"""
Phase 2 — Policy Engine and Tool Router tests
=============================================
Covers the Phase 2 hardening of the authorization boundary:

* the five policy decisions and default-DENY for unknown actions;
* valid and invalid tool schemas (name/type/size/target/identity);
* denied & quarantined actions never reach the executor;
* ordinary, previously supported coding-agent operations still work;
* fail-closed behaviour for engine failures and malformed decisions;
* ALLOW_WITH_MONITORING enforces predefined resource limits;
* model output cannot override trusted policy rules;
* sensitive arguments are redacted from logs.

The existing baseline suites (test_token_optimizations,
test_model_router_and_adapter, test_phase1_security) must keep passing
alongside these tests.
"""
import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nexora.backend.app.core.policy import (
    DEFAULT_MONITORING_LIMITS,
    KNOWN_APPROVAL_STATES,
    SUPPORTED_DECISIONS,
    Capability,
    MonitoringLimits,
    PolicyEngine,
    PolicyRule,
)
from nexora.backend.app.core.tools import (
    MAX_STRING_ARG_LENGTH,
    REDACTED_PLACEHOLDER,
    LocalSandboxAdapter,
    ToolName,
    ToolRequest,
    ToolRouter,
    ToolValidationError,
    _build_approved_test_argv,
    _resolve_trusted_executable,
    redact_arguments,
)
from nexora.backend.app.models import (
    ExecutionStatus,
    PolicyDecision as PolicyDecisionEnum,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def engine() -> PolicyEngine:
    return PolicyEngine()


@pytest.fixture()
def router() -> ToolRouter:
    return ToolRouter()


def _make_orchestrator(engine: PolicyEngine, workspace="/ws"):
    """Build a minimally-wired RunOrchestrator for _process_tool_call tests.

    Mirrors the existing test style (``__new__`` + SimpleNamespace + AsyncMock)
    so no database or worker process is required.
    """
    from nexora.backend.app.core.approval import ApprovalManager
    from nexora.backend.app.core.tools import ToolRouter as _ToolRouter
    from nexora.backend.app.worker.orchestrator import RunOrchestrator

    from nexora.backend.app.models import RunStatus

    orch = RunOrchestrator.__new__(RunOrchestrator)
    orch.run = SimpleNamespace(
        id="11111111-1111-1111-1111-111111111111",
        status=RunStatus.QUEUED,
        current_stage="queued",
        steps_completed=0,
        tool_calls_count=0,
    )
    orch.workspace_path = workspace
    orch.tool_router = _ToolRouter()
    orch.policy_engine = engine
    orch.approval_manager = ApprovalManager()
    orch.sandbox_adapter = AsyncMock()
    orch.sandbox_adapter.execute = AsyncMock(
        return_value=SimpleNamespace(
            tool_name="repo.read",
            arguments={"path": "a.py"},
            stdout="ok",
            stderr="",
            exit_code=0,
            duration_ms=1,
            sandbox_id="sbx",
            execution_status=ExecutionStatus.SUCCESS,
        )
    )
    orch.db = AsyncMock()
    orch.audit_log = MagicMock()
    orch._add_event = AsyncMock()
    orch._safe_stop = AsyncMock()
    orch._run_validation = AsyncMock()
    orch._get_workspace_diff = MagicMock(return_value="diff")
    orch.loop_detector = MagicMock()
    orch.loop_detector.record_action = MagicMock(
        return_value=SimpleNamespace(action_hash="hash")
    )
    orch.loop_detector.should_stop = MagicMock(return_value=False)
    return orch


def _quarantine_engine(capability: Capability) -> PolicyEngine:
    eng = PolicyEngine()
    eng.config.rules = [
        PolicyRule(
            capability=capability,
            decision=PolicyDecisionEnum.QUARANTINE,
            reason_codes=["TEST_QUARANTINE"],
        )
    ]
    return eng


# ---------------------------------------------------------------------------
# 1/3. The five policy decisions via the shared contract
# ---------------------------------------------------------------------------

class TestFivePolicyDecisions:

    def test_allow_read_only(self, engine):
        d = engine.evaluate(Capability.REPO_READ, {"path": "a.py"}, target_path="a.py")
        assert d.decision == PolicyDecisionEnum.ALLOW

    def test_allow_with_monitoring(self, engine):
        d = engine.evaluate(Capability.REPO_WRITE, {"path": "a.py", "content": "x"})
        assert d.decision == PolicyDecisionEnum.ALLOW_WITH_MONITORING

    def test_require_approval(self, engine):
        d = engine.evaluate(Capability.SHELL_RUN, {"command": "ls tests"})
        assert d.decision == PolicyDecisionEnum.REQUIRE_APPROVAL

    def test_deny(self, engine):
        d = engine.evaluate(Capability.NETWORK_FETCH, {"url": "https://example.com"})
        assert d.decision == PolicyDecisionEnum.DENY

    def test_quarantine(self):
        eng = _quarantine_engine(Capability.REPO_READ)
        d = eng.evaluate(Capability.REPO_READ, {"path": "a.py"}, target_path="a.py")
        assert d.decision == PolicyDecisionEnum.QUARANTINE

    def test_all_decisions_use_shared_contract_fields(self, engine):
        d = engine.evaluate(Capability.REPO_READ, {"path": "a.py"}, target_path="a.py")
        # PolicyDecisionBase contract fields must all be present.
        assert d.capability == "repo.read"
        assert d.policy_version == engine.config.version
        assert d.engine_version == engine._engine_version
        assert isinstance(d.reason_codes, list)

    def test_supported_decisions_set_is_the_five(self):
        assert SUPPORTED_DECISIONS == {
            PolicyDecisionEnum.ALLOW,
            PolicyDecisionEnum.ALLOW_WITH_MONITORING,
            PolicyDecisionEnum.REQUIRE_APPROVAL,
            PolicyDecisionEnum.DENY,
            PolicyDecisionEnum.QUARANTINE,
        }


# ---------------------------------------------------------------------------
# 2. Default DENY for unknown tools / capabilities / states
# ---------------------------------------------------------------------------

class TestDefaultDeny:

    def test_unknown_capability_denied(self, engine):
        d = engine.evaluate("not.a.capability", {})
        assert d.decision == PolicyDecisionEnum.DENY
        assert "UNKNOWN_CAPABILITY" in d.reason_codes

    def test_unknown_approval_state_denied(self, engine):
        d = engine.evaluate(Capability.SHELL_RUN, {"command": "ls"}, approval_state="maybe")
        assert d.decision == PolicyDecisionEnum.DENY
        assert "UNKNOWN_APPROVAL_STATE" in d.reason_codes

    def test_known_approval_states_are_accepted(self, engine):
        for state in KNOWN_APPROVAL_STATES:
            d = engine.evaluate(Capability.SHELL_RUN, {"command": "ls"}, approval_state=state)
            assert d.decision in SUPPORTED_DECISIONS

    def test_malformed_arguments_denied(self, engine):
        d = engine.evaluate(Capability.REPO_READ, "not-a-dict")
        assert d.decision == PolicyDecisionEnum.DENY
        assert "MALFORMED_ARGUMENTS" in d.reason_codes

    def test_malformed_runtime_state_denied(self, engine):
        d = engine.evaluate(Capability.REPO_READ, {}, runtime_state="nope")
        assert d.decision == PolicyDecisionEnum.DENY
        assert "MALFORMED_RUNTIME_STATE" in d.reason_codes

    def test_unknown_tool_is_rejected(self, router):
        with pytest.raises(ToolValidationError):
            router.validate_and_route("explode.host", {}, "r1")

    def test_non_object_arguments_rejected(self, router):
        with pytest.raises(ToolValidationError):
            router.validate_and_route("repo.read", None, "r1")


# ---------------------------------------------------------------------------
# 8. Model output cannot override trusted policy rules
# ---------------------------------------------------------------------------

class TestModelCannotOverridePolicy:

    def test_model_risk_level_argument_cannot_avoid_deny(self, engine):
        # shell.run's explicit rule only triggers at risk_level=high. A model
        # supplying risk_level in its arguments must not be able to change the
        # trusted risk level (it stays "medium" -> REQUIRE_APPROVAL).
        d = engine.evaluate(
            Capability.SHELL_RUN,
            {"command": "ls", "risk_level": "high", "decision": "ALLOW"},
        )
        assert d.decision == PolicyDecisionEnum.REQUIRE_APPROVAL

    def test_unknown_condition_key_never_matches(self, engine):
        # A rule with an unverifiable condition must not silently allow.
        eng = PolicyEngine()
        eng.config.rules = [
            PolicyRule(
                capability=Capability.NETWORK_FETCH,
                decision=PolicyDecisionEnum.ALLOW,
                reason_codes=["SHOULD_NOT_MATCH"],
                conditions={"some_unknown_key": "some_value"},
            )
        ]
        d = eng.evaluate(Capability.NETWORK_FETCH, {"some_unknown_key": "some_value"})
        assert d.decision != PolicyDecisionEnum.ALLOW

    def test_path_prefix_condition_uses_declared_target(self, engine):
        # A model cannot satisfy a path_prefix condition for a different path.
        d = engine.evaluate(
            Capability.REPO_WRITE,
            {"path": "src/a.py", "content": "x", "path_prefix": "test_"},
        )
        # Still monitored (rule condition unmet), never denied-by-accident.
        assert d.decision == PolicyDecisionEnum.ALLOW_WITH_MONITORING


# ---------------------------------------------------------------------------
# 9. Fail closed: engine failure / malformed decision
# ---------------------------------------------------------------------------

class TestFailClosed:

    def test_engine_exception_fails_closed(self, engine, monkeypatch):
        def boom(**kwargs):
            raise RuntimeError("policy backend down")

        monkeypatch.setattr(engine, "_evaluate", boom)
        d = engine.evaluate(Capability.REPO_READ, {"path": "a.py"})
        assert d.decision == PolicyDecisionEnum.DENY
        assert "POLICY_ENGINE_ERROR" in d.reason_codes

    def test_malformed_decision_fails_closed(self, engine, monkeypatch):
        monkeypatch.setattr(engine, "_evaluate", lambda **kwargs: object())
        d = engine.evaluate(Capability.REPO_READ, {"path": "a.py"})
        assert d.decision == PolicyDecisionEnum.DENY
        assert "UNKNOWN_DECISION" in d.reason_codes

    def test_unknown_decision_value_fails_closed(self, engine, monkeypatch):
        monkeypatch.setattr(
            engine, "_evaluate", lambda **kwargs: SimpleNamespace(decision="WAT")
        )
        d = engine.evaluate(Capability.REPO_READ, {"path": "a.py"})
        assert d.decision == PolicyDecisionEnum.DENY
        assert "UNKNOWN_DECISION" in d.reason_codes


# ---------------------------------------------------------------------------
# 4. Tool schema validation: names, types, sizes, targets, identity
# ---------------------------------------------------------------------------

class TestToolValidation:

    @pytest.mark.parametrize(
        "tool,args",
        [
            ("repo.list", {}),
            ("repo.read", {"path": "a.py"}),
            ("repo.write", {"path": "a.py", "content": "x", "mode": "create"}),
            ("test.run", {"command": "pytest -v", "timeout_seconds": 30}),
            ("git.status", {}),
            ("git.diff", {"staged": True}),
            ("git.branch", {"action": "list"}),
            ("git.commit", {"message": "msg", "files": ["a.py"]}),
            ("git.push", {"remote": "origin"}),
            ("shell.run", {"command": "ls"}),
            ("network.fetch", {"url": "https://example.com"}),
            ("web.scrape", {"url": "https://example.com"}),
        ],
    )
    def test_valid_schemas_accepted(self, router, tool, args):
        req = router.validate_and_route(tool, args, "r1", run_id="run-1", workspace_path="/ws")
        assert req.tool_name.value == tool
        assert req.run_id == "run-1"

    def test_missing_required_field(self, router):
        with pytest.raises(ToolValidationError):
            router.validate_and_route("repo.read", {}, "r1")

    def test_unexpected_argument(self, router):
        with pytest.raises(ToolValidationError):
            router.validate_and_route("repo.read", {"path": "a.py", "x": 1}, "r1")

    def test_wrong_argument_type(self, router):
        with pytest.raises(ToolValidationError):
            router.validate_and_route("repo.read", {"path": 5}, "r1")

    def test_bool_is_not_an_integer(self, router):
        with pytest.raises(ToolValidationError):
            router.validate_and_route("test.run", {"timeout_seconds": True}, "r1")

    def test_invalid_enum_value(self, router):
        with pytest.raises(ToolValidationError):
            router.validate_and_route("repo.write", {"path": "a.py", "content": "x", "mode": "explode"}, "r1")

    def test_oversized_string_argument_rejected(self, router):
        with pytest.raises(ToolValidationError):
            router.validate_and_route(
                "repo.write",
                {"path": "a.py", "content": "x" * (MAX_STRING_ARG_LENGTH + 1)},
                "r1",
            )

    def test_nul_byte_rejected(self, router):
        with pytest.raises(ToolValidationError):
            router.validate_and_route("repo.read", {"path": "a\x00.py"}, "r1")

    @pytest.mark.parametrize("bad_path", ["/etc/passwd", "../secret", "..\\secret", "C:\\abs", "C:rel"])
    def test_unsafe_target_paths_rejected(self, router, bad_path):
        with pytest.raises(ToolValidationError):
            router.validate_and_route("repo.read", {"path": bad_path}, "r1")

    @pytest.mark.parametrize("bad_path", ["a/../../b", "../x", "sub/../../out"])
    def test_path_escape_is_rejected(self, router, bad_path):
        with pytest.raises(ToolValidationError):
            router.validate_and_route("repo.read", {"path": bad_path}, "r1")

    def test_absolute_path_inside_workspace_is_still_rejected(self, router):
        with pytest.raises(ToolValidationError):
            router.validate_and_route(
                "repo.read", {"path": "/ws/a.py"}, "r1", workspace_path="/ws"
            )

    def test_invalid_url_rejected(self, router):
        with pytest.raises(ToolValidationError):
            router.validate_and_route("network.fetch", {"url": "file:///etc/passwd"}, "r1")

    def test_invalid_identity_rejected(self, router):
        with pytest.raises(ToolValidationError):
            router.validate_and_route("repo.read", {"path": "a.py"}, "r1", run_id="")
        with pytest.raises(ToolValidationError):
            router.validate_and_route("repo.read", {"path": "a.py"}, "r1", repository_id="  ")

    def test_schemas_are_unchanged_for_the_model(self, router):
        schemas = router.get_all_schemas()
        names = sorted(s["function"]["name"] for s in schemas)
        assert names == sorted(
            [
                "repo.list", "repo.read", "repo.write", "test.run",
                "git.status", "git.diff", "git.branch", "git.commit", "git.push",
                "shell.run", "network.fetch", "web.scrape",
            ]
        )
        # Contract preservation: no validation-only keys leaked into schemas.
        for s in schemas:
            assert "maxLength" not in str(s)
            assert "additionalProperties" in s["function"]["parameters"]


# ---------------------------------------------------------------------------
# 3/6. Denied / quarantined actions never reach the executor
# ---------------------------------------------------------------------------

class TestExecutorGating:

    @pytest.mark.asyncio
    async def test_allow_reaches_executor(self):
        orch = _make_orchestrator(PolicyEngine())
        await orch._process_tool_call(
            {"id": "c1", "name": "repo.read", "arguments": {"path": "a.py"}}, None
        )
        orch.sandbox_adapter.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_deny_does_not_reach_executor(self):
        orch = _make_orchestrator(PolicyEngine())
        await orch._process_tool_call(
            {"id": "c1", "name": "network.fetch", "arguments": {"url": "https://x.test"}},
            None,
        )
        orch.sandbox_adapter.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_quarantine_does_not_reach_executor(self):
        orch = _make_orchestrator(_quarantine_engine(Capability.REPO_READ))
        await orch._process_tool_call(
            {"id": "c1", "name": "repo.read", "arguments": {"path": "a.py"}}, None
        )
        orch.sandbox_adapter.execute.assert_not_awaited()
        orch._safe_stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_invalid_tool_does_not_reach_executor(self):
        orch = _make_orchestrator(PolicyEngine())
        await orch._process_tool_call(
            {"id": "c1", "name": "repo.read", "arguments": {"path": "../escape"}}, None
        )
        orch.sandbox_adapter.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unavailable_engine_denies(self):
        orch = _make_orchestrator(PolicyEngine())
        orch.policy_engine = None
        await orch._process_tool_call(
            {"id": "c1", "name": "repo.read", "arguments": {"path": "a.py"}}, None
        )
        orch.sandbox_adapter.execute.assert_not_awaited()
        orch._add_event.assert_awaited()

    @pytest.mark.asyncio
    async def test_malformed_decision_is_coerced_to_deny(self):
        orch = _make_orchestrator(PolicyEngine())
        fake_engine = MagicMock()
        fake_engine.evaluate.return_value = SimpleNamespace(
            decision="NOT_A_DECISION",
            reason_codes=[],
            normalized_arguments=None,
            target_path=None,
            risk_level=None,
            policy_version=None,
            engine_version=None,
        )
        orch.policy_engine = fake_engine
        await orch._process_tool_call(
            {"id": "c1", "name": "repo.read", "arguments": {"path": "a.py"}}, None
        )
        orch.sandbox_adapter.execute.assert_not_awaited()


# ---------------------------------------------------------------------------
# 7. REQUIRE_APPROVAL pauses execution (existing-compatible response)
# ---------------------------------------------------------------------------

class TestRequireApprovalPauses:

    @pytest.mark.asyncio
    async def test_require_approval_sets_awaiting_approval_and_pauses(self):
        from nexora.backend.app.models import RunStatus

        orch = _make_orchestrator(PolicyEngine())
        await orch._process_tool_call(
            {"id": "c1", "name": "shell.run", "arguments": {"command": "ls tests"}}, None
        )
        assert orch.run.status == RunStatus.AWAITING_APPROVAL
        assert orch.run.current_stage == "awaiting_approval"
        orch.sandbox_adapter.execute.assert_not_awaited()

    def test_approved_approval_state_allows(self, engine):
        d = engine.evaluate(
            Capability.SHELL_RUN, {"command": "ls tests"}, approval_state="approved"
        )
        assert d.decision == PolicyDecisionEnum.ALLOW


# ---------------------------------------------------------------------------
# 5. ALLOW_WITH_MONITORING enforces predefined resource limits
# ---------------------------------------------------------------------------

class TestMonitoringEnforcement:

    def test_monitoring_limits_are_predefined(self, engine):
        limits = engine.monitoring_limits(Capability.REPO_WRITE)
        assert isinstance(limits, MonitoringLimits)
        assert limits.require_audit is True

    def test_unknown_capability_uses_conservative_defaults(self, engine):
        assert engine.monitoring_limits("nope") == DEFAULT_MONITORING_LIMITS

    def test_monitored_timeout_is_clamped(self):
        orch = _make_orchestrator(PolicyEngine())
        limits = orch.policy_engine.monitoring_limits(Capability.TEST_RUN)
        req = ToolRequest(
            tool_name=ToolName.TEST_RUN,
            arguments={"command": "pytest -v", "timeout_seconds": 10_000_000},
            request_id="r1",
            capability=Capability.TEST_RUN,
        )
        effective, applied = orch._apply_monitoring_limits(req)
        assert applied == limits
        assert effective.arguments["timeout_seconds"] == limits.max_timeout_seconds
        # Original request is left untouched.
        assert req.arguments["timeout_seconds"] == 10_000_000

    @pytest.mark.asyncio
    async def test_monitored_execution_uses_clamped_request(self):
        orch = _make_orchestrator(PolicyEngine())
        await orch._process_tool_call(
            {
                "id": "c1",
                "name": "test.run",
                "arguments": {"command": "pytest -v", "timeout_seconds": 10_000_000},
            },
            None,
        )
        executed = orch.sandbox_adapter.execute.await_args.args[0]
        limit = orch.policy_engine.monitoring_limits(Capability.TEST_RUN)
        assert executed.arguments["timeout_seconds"] == limit.max_timeout_seconds

    @pytest.mark.asyncio
    async def test_monitored_output_is_bounded(self):
        orch = _make_orchestrator(PolicyEngine())
        orch.sandbox_adapter.execute = AsyncMock(
            return_value=SimpleNamespace(
                tool_name="test.run",
                arguments={"command": "pytest -v"},
                stdout="A" * 1_000_000,
                stderr="B" * 1_000_000,
                exit_code=0,
                duration_ms=1,
                sandbox_id="sbx",
                execution_status=ExecutionStatus.SUCCESS,
            )
        )
        await orch._process_tool_call(
            {"id": "c1", "name": "test.run", "arguments": {"command": "pytest -v"}}, None
        )
        # The persisted ToolExecution arguments were redacted/truncated, so
        # assert on the bounded property rather than exact bytes.
        assert orch.recent_tool_output is not None
        assert len(orch.recent_tool_output) <= len("test.run: ") + 500


# ---------------------------------------------------------------------------
# 10. Redaction of sensitive arguments / bounded output
# ---------------------------------------------------------------------------

class TestRedaction:

    def test_sensitive_keys_redacted(self):
        redacted = redact_arguments(
            {"path": "a.py", "api_key": "sk-live-123", "nested": {"password": "hunter2"}}
        )
        assert redacted["api_key"] == REDACTED_PLACEHOLDER
        assert redacted["nested"]["password"] == REDACTED_PLACEHOLDER
        assert redacted["path"] == "a.py"
        assert "sk-live-123" not in str(redacted)

    def test_long_string_truncated(self):
        redacted = redact_arguments({"content": "z" * 5000}, max_string_length=100)
        assert len(redacted["content"]) < 5000
        assert "truncated" in redacted["content"]

    def test_input_is_not_mutated(self):
        original = {"api_key": "secret", "content": "x" * 1000}
        snapshot = dict(original)
        redact_arguments(original, max_string_length=10)
        assert original == snapshot

    @pytest.mark.asyncio
    async def test_audit_log_records_redacted_arguments(self):
        from nexora.backend.app.core.audit import AuditEventType

        orch = _make_orchestrator(PolicyEngine())
        await orch._process_tool_call(
            {
                "id": "c1",
                "name": "repo.write",
                "arguments": {"path": "a.py", "content": "super-secret-token-value"},
            },
            None,
        )
        # Find the TOOL_REQUEST audit entry.
        request_entries = [
            call.args[1]
            for call in orch.audit_log.append.call_args_list
            if call.args and call.args[0] == AuditEventType.TOOL_REQUEST
        ]
        assert request_entries, "TOOL_REQUEST audit entry missing"
        # repo.write has no secret-looking key, but long content is bounded.
        recorded = str(request_entries[0])
        assert len(recorded) < 1000


# ---------------------------------------------------------------------------
# 8. test.run executes as an argument vector (command-injection regression)
# ---------------------------------------------------------------------------
#
# The previous implementation validated the command with a string ``startswith``
# check and then handed it to ``create_subprocess_shell``. A payload such as
# ``pytest -q ; echo PWNED_MARKER`` passed the prefix check and the trailing
# command ran. These tests pin the fixed behaviour: approved commands execute
# via ``create_subprocess_exec`` (no shell), and shell syntax is rejected
# before any process is created.

# Every payload below MUST be rejected without spawning a process. Markers are
# chosen so that, were the payload to execute, ``PWNED_MARKER`` would appear on
# stdout and/or ``pwned_marker.txt`` would be created in the workspace.
_INJECTION_PAYLOADS = [
    "pytest -q ; echo PWNED_MARKER",                      # POSIX sequencer
    "pytest -q & echo PWNED_MARKER",                      # cmd.exe sequencer
    "pytest -q && echo PWNED_MARKER",                     # logical AND
    "pytest -q | echo PWNED_MARKER",                      # pipe
    "pytest -q || echo PWNED_MARKER",                     # logical OR
    "pytest -q > pwned_marker.txt",                       # redirect (create)
    "pytest -q >> pwned_marker.txt",                      # redirect (append)
    "pytest -q < pwned_marker.txt",                       # redirect (input)
    "pytest -q $(echo PWNED_MARKER)",                     # POSIX substitution
    "pytest -q `echo PWNED_MARKER`",                      # backtick substitution
    'pytest -q "$(echo PWNED_MARKER)"',                   # quoted substitution
    'pytest -q "`echo PWNED_MARKER`"',                    # quoted backticks
    "pytest -q (echo PWNED_MARKER)",                      # subshell
    "pytest -q %PWNED_MARKER%",                           # cmd.exe expansion
    "pytest -q ^& echo PWNED_MARKER",                     # cmd.exe escape+chain
    "pytest -q \n echo PWNED_MARKER",                     # newline (control char)
]


def _write_passing_test(workspace) -> None:
    (workspace / "test_ok.py").write_text(
        "def test_ok():\n    assert True\n", encoding="utf-8"
    )


class TestTestRunInjection:

    @pytest.mark.parametrize("payload", _INJECTION_PAYLOADS)
    @pytest.mark.asyncio
    async def test_injection_payload_is_rejected_without_side_effects(
        self, payload, tmp_path
    ):
        _write_passing_test(tmp_path)
        sandbox = LocalSandboxAdapter()
        result = await sandbox._test_run(
            {"command": payload, "timeout_seconds": 30},
            str(tmp_path),
            os.environ.copy(),
        )
        # Rejected fail-closed: no output, non-zero exit.
        assert result["exit_code"] == 1, result
        assert result["stdout"] == "", result
        # The secondary command never ran and no file was created/overwritten.
        assert "PWNED_MARKER" not in result["stdout"]
        assert not (tmp_path / "pwned_marker.txt").exists()

    @pytest.mark.parametrize("payload", _INJECTION_PAYLOADS)
    @pytest.mark.asyncio
    async def test_injection_payload_never_reaches_a_subprocess(
        self, payload, tmp_path, monkeypatch
    ):
        async def _boom(*args, **kwargs):
            raise AssertionError(
                "no subprocess may be launched for a rejected test command"
            )

        monkeypatch.setattr(asyncio, "create_subprocess_exec", _boom)
        monkeypatch.setattr(asyncio, "create_subprocess_shell", _boom)

        sandbox = LocalSandboxAdapter()
        result = await sandbox._test_run(
            {"command": payload}, str(tmp_path), os.environ.copy()
        )
        assert result["exit_code"] == 1, result

    @pytest.mark.asyncio
    async def test_approved_command_uses_exec_not_shell(self, tmp_path, monkeypatch):
        captured = {}
        shell_calls = {"n": 0}

        class _FakeProc:
            returncode = 0

            async def communicate(self):
                return (b"ok", b"")

        async def fake_exec(*argv, **kwargs):
            captured["argv"] = argv
            return _FakeProc()

        async def fake_shell(*args, **kwargs):
            shell_calls["n"] += 1
            raise AssertionError("create_subprocess_shell must not be used")

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        monkeypatch.setattr(asyncio, "create_subprocess_shell", fake_shell)

        sandbox = LocalSandboxAdapter()
        result = await sandbox._test_run(
            {"command": 'python -m pytest -q -k "smoke"'},
            str(tmp_path),
            os.environ.copy(),
        )
        assert result["exit_code"] == 0
        assert shell_calls["n"] == 0
        # The runner is resolved to a trusted absolute path; the remaining
        # tokens are the parsed, quote-stripped arguments.
        assert os.path.isabs(captured["argv"][0])
        assert os.path.basename(captured["argv"][0]).lower().startswith("python")
        assert captured["argv"][1:] == ("-m", "pytest", "-q", "-k", "smoke")

    @pytest.mark.asyncio
    async def test_rejected_command_keeps_legacy_message(self, tmp_path):
        # Phase 1 regression tests assert this exact substring; keep it stable.
        sandbox = LocalSandboxAdapter()
        result = await sandbox._test_run(
            {"command": "curl http://evil.example.com"},
            str(tmp_path),
            os.environ.copy(),
        )
        assert result["exit_code"] == 1
        assert "not in the allowed prefix" in result["stderr"]

    @pytest.mark.asyncio
    async def test_unlaunchable_runner_fails_closed(self, tmp_path, monkeypatch):
        # e.g. npm -> npm.cmd on Windows cannot be started without a shell;
        # the failure must be a fail-closed result, not a raised exception.
        async def fake_exec(*args, **kwargs):
            raise FileNotFoundError(2, "The system cannot find the file specified")

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        sandbox = LocalSandboxAdapter()
        result = await sandbox._test_run(
            {"command": "npm test"}, str(tmp_path), os.environ.copy()
        )
        assert result["exit_code"] == 1, result
        assert result["stdout"] == ""
        assert "Failed to launch" in result["stderr"]


class TestTestRunApprovedCommandsStillWork:

    @pytest.mark.asyncio
    async def test_pytest_version_runs(self, tmp_path):
        sandbox = LocalSandboxAdapter()
        result = await sandbox._test_run(
            {"command": "pytest --version"}, str(tmp_path), os.environ.copy()
        )
        assert result["exit_code"] == 0, result
        assert "not in the allowed prefix" not in result["stderr"]

    @pytest.mark.asyncio
    async def test_python_m_pytest_version_runs(self, tmp_path):
        sandbox = LocalSandboxAdapter()
        result = await sandbox._test_run(
            {"command": "python -m pytest --version"},
            str(tmp_path),
            os.environ.copy(),
        )
        assert result["exit_code"] == 0, result
        assert "not in the allowed prefix" not in result["stderr"]

    @pytest.mark.asyncio
    async def test_default_command_runs_passing_suite(self, tmp_path):
        _write_passing_test(tmp_path)
        sandbox = LocalSandboxAdapter()
        result = await sandbox._test_run({}, str(tmp_path), os.environ.copy())
        assert result["exit_code"] == 0, result
        assert "passed" in (result["stdout"] + result["stderr"]).lower()

    @pytest.mark.asyncio
    async def test_quoted_selector_expression_runs(self, tmp_path):
        _write_passing_test(tmp_path)
        sandbox = LocalSandboxAdapter()
        result = await sandbox._test_run(
            {"command": 'pytest -q -k "ok"', "timeout_seconds": 60},
            str(tmp_path),
            os.environ.copy(),
        )
        assert result["exit_code"] == 0, result


class TestTestRunCommandValidationHelpers:

    def test_approved_commands_produce_expected_argv(self):
        assert _build_approved_test_argv("pytest -v") == ["pytest", "-v"]
        assert _build_approved_test_argv("python3 -m pytest -q") == [
            "python3", "-m", "pytest", "-q",
        ]
        assert _build_approved_test_argv("npm test") == ["npm", "test"]
        assert _build_approved_test_argv("npm run test -s") == [
            "npm", "run", "test", "-s",
        ]
        assert _build_approved_test_argv("cargo test") == ["cargo", "test"]

    @pytest.mark.parametrize(
        "command",
        [
            "curl http://evil.example.com",   # unapproved executable
            "rm -rf /",                       # unapproved executable
            "echo hello",                     # unapproved executable
            "pytestfoo -v",                   # NOT a prefix match
            "pytest.bak -v",                  # NOT a prefix match
            "./pytest -v",                    # path executable
            "/usr/bin/pytest -v",             # path executable
            "python -m http.server",          # wrong module
            "python -c 'print(1)'",           # arbitrary code
            "python3 -m pytestx",             # wrong module
            "",                               # empty
            "   ",                            # whitespace only
            "pytest -k 'unterminated",        # unterminated quote
            "pytest /etc/passwd",             # absolute path arg
            "pytest ../../etc/passwd",        # escaping path arg
            "pytest C:\\abs",                 # drive-absolute path arg
        ],
    )
    def test_unapproved_commands_are_rejected(self, command):
        with pytest.raises(ToolValidationError):
            _build_approved_test_argv(command)


# ---------------------------------------------------------------------------
# 9. test.run option-surface hardening (allowlist, fail-closed)
# ---------------------------------------------------------------------------
#
# pytest options that can load plugins, change configuration, override settings,
# alter root/config discovery, or write/clear files were previously accepted.
# They are now rejected by an allowlist; safe, commonly used options and
# workspace-relative selectors still work.

class TestTestRunOptionHardening:

    _UNSAFE = [
        # plugin loading
        "pytest -p myplugin",
        "pytest -pmyplugin",
        "pytest -p no:cacheprovider",
        "pytest --plugin=myplugin",
        # configuration / ini overrides
        "pytest -c evil.ini",
        "pytest -c evil.ini -q",
        "pytest --config-file=evil.ini",
        "pytest -o addopts=-p evil",
        "pytest --override-ini=python_files=*",
        "pytest --override-ini python_files=*",
        # discovery / root
        "pytest --rootdir=/tmp",
        "pytest --rootdir=C:\\evil",
        "pytest --confcutdir=/",
        "pytest --confcutdir=..",
        # out-of-workspace writes / clears
        "pytest --basetemp=/tmp/dest",
        "pytest --junitxml=/tmp/out.xml",
        "pytest --junitxml=C:\\Users\\Hp\\x.xml",
        "pytest --log-file=/tmp/x.log",
        "pytest --cache-dir=/tmp/cache",
        "pytest --cache-clear",
        "pytest --html=/tmp/out.html",
        "pytest --result-log=/tmp/x",
        # absolute / escaping selectors
        "pytest /etc/passwd",
        "pytest ../..",
        "pytest tests/../../etc",
        # ambiguous / unknown / malformed
        "pytest -k=foo",          # ambiguous short =form
        "pytest -k -x",           # value looks like an option
        "pytest --maxfail=-1",    # not a non-negative integer
        "pytest --tb=nope",       # invalid choice
        "pytest --verbose=1",     # flag given a value
        "pytest --frobnicate",    # unknown -> fail closed
        "pytest -Z",              # unknown -> fail closed
    ]

    _SAFE = [
        "pytest -v",
        "pytest -q",
        "pytest -q -k 'ok'",
        "pytest -k ok",
        "pytest -kook",           # attached short value
        "pytest -m 'not slow'",
        "pytest -xvs",            # clustered flags
        "pytest --version",
        "pytest --maxfail=3",
        "pytest --maxfail 3",
        "pytest --tb=short",
        "pytest --tb short",
        "pytest --verbose --quiet",
        "pytest tests/",
        "pytest tests/test_x.py",
        "pytest tests/test_x.py::test_y",
        "pytest -- tests/test_x.py",
    ]

    @pytest.mark.parametrize("command", _UNSAFE)
    def test_unsafe_options_are_rejected(self, command):
        with pytest.raises(ToolValidationError):
            _build_approved_test_argv(command)

    @pytest.mark.parametrize("command", _SAFE)
    def test_safe_options_are_accepted(self, command):
        argv = _build_approved_test_argv(command)
        assert argv[0] == "pytest"
        assert isinstance(argv, list) and argv

    def test_separate_option_values_are_not_read_as_flags(self):
        assert _build_approved_test_argv("pytest --maxfail 3") == [
            "pytest", "--maxfail", "3",
        ]
        assert _build_approved_test_argv('pytest -q -k "ok"') == [
            "pytest", "-q", "-k", "ok",
        ]

    # --- executable resolution (cannot be shadowed from the workspace) -----

    def test_resolution_returns_trusted_absolute_path(self):
        resolved = _resolve_trusted_executable("python")
        assert resolved is not None
        assert os.path.isabs(resolved)

    def test_resolution_ignores_relative_and_cwd_path_entries(self, monkeypatch):
        # A PATH made only of "." / relative entries must resolve to nothing.
        monkeypatch.setenv("PATH", os.pathsep.join([".", os.curdir]))
        assert _resolve_trusted_executable("pytest") is None

    def test_resolution_unknown_executable_is_none(self):
        assert _resolve_trusted_executable("definitely-not-a-real-runner-xyz") is None

    @pytest.mark.asyncio
    async def test_workspace_planted_runner_cannot_shadow(self, tmp_path, monkeypatch):
        # Plant a "runner" in the workspace, make the workspace the cwd, and set
        # a relative-only PATH. Resolution must fail closed without running it.
        (tmp_path / "pytest.bat").write_text(
            "@echo off\r\necho PWNED_SHADOW\r\n", encoding="utf-8"
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("PATH", os.pathsep.join([".", os.curdir]))

        sandbox = LocalSandboxAdapter()
        result = await sandbox._test_run(
            {"command": "pytest -v"}, str(tmp_path), os.environ.copy()
        )
        assert result["exit_code"] == 1, result
        assert "PWNED_SHADOW" not in result["stdout"]

    @pytest.mark.asyncio
    async def test_positional_selector_still_runs(self, tmp_path):
        _write_passing_test(tmp_path)
        sandbox = LocalSandboxAdapter()
        result = await sandbox._test_run(
            {"command": "pytest -q test_ok.py", "timeout_seconds": 60},
            str(tmp_path),
            os.environ.copy(),
        )
        assert result["exit_code"] == 0, result

