"""
Phase 4 — Secure backend integration & approval workflow tests
==============================================================
Covers the migration-free Phase 4 hardening of the authorization and
approval boundary:

* explicit ``REQUIRE_APPROVAL`` policy rules honour a verified ``approved``
  state (the confirmed ``git.commit`` approval-rule deadlock);
* ``DENY`` / ``QUARANTINE`` remain terminal and can never be overridden by an
  approval;
* the approval-key mismatch is fixed: approvals are stored and looked up by
  one consistent, server-owned identifier (the tool request id);
* the approval lifecycle: binding to run + scope fingerprint, expiry,
  single-use consumption, replay rejection, wrong-run rejection, and
  argument-change invalidation;
* ``REQUIRE_APPROVAL`` pauses execution and never reaches the sandbox;
* the ``/runs/{id}/approve`` route no longer hard-codes an approver identity,
  enforces run ownership, maps invalid transitions to safe errors, and records
  a correlated run event.

Authentication and per-user ownership are intentionally out of scope for this
change set (no authentication architecture exists; see the Phase 4 report).
Those are covered separately / reported as blocked.

The existing Phase 1-3 suites must keep passing alongside these tests.
"""
import inspect
import uuid
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from nexora.backend.app.core.approval import (
    ApprovalError,
    ApprovalManager,
    ApprovalRequest,
    fingerprint_scope,
)
from nexora.backend.app.core.policy import Capability, PolicyEngine, PolicyRule
from nexora.backend.app.core.tools import ToolRouter
from nexora.backend.app.models import (
    ApprovalStatus,
    ExecutionStatus,
    PolicyDecision as PolicyDecisionEnum,
    RunStatus,
)
from nexora.backend.app.schemas import ApproveActionRequest


RUN_ID = "11111111-1111-1111-1111-111111111111"
OTHER_RUN_ID = "22222222-2222-2222-2222-222222222222"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeSession:
    """Minimal async session that assigns a real id on flush.

    The orchestrator relies on ``flush()`` to materialise primary keys, so a
    plain ``AsyncMock`` would leave every id as ``None`` (making the approval
    key binding untestable). This fake mirrors the real behaviour closely
    enough to exercise the identifier plumbing.
    """

    def __init__(self):
        self.added = []

    def add(self, obj):
        self.added.append(obj)

    async def flush(self):
        for obj in self.added:
            if getattr(obj, "id", None) is None:
                try:
                    obj.id = str(uuid.uuid4())
                except Exception:
                    pass

    async def commit(self):  # pragma: no cover - not used by _process_tool_call
        pass


def _make_orchestrator(engine=None, workspace="/ws"):
    """Build a minimally-wired RunOrchestrator (no DB, no worker process)."""
    from nexora.backend.app.worker.orchestrator import RunOrchestrator

    orch = RunOrchestrator.__new__(RunOrchestrator)
    orch.run = SimpleNamespace(
        id=RUN_ID,
        status=RunStatus.QUEUED,
        current_stage="queued",
        steps_completed=0,
        tool_calls_count=0,
    )
    orch.workspace_path = workspace
    orch.tool_router = ToolRouter()
    orch.policy_engine = engine or PolicyEngine()
    orch.approval_manager = ApprovalManager()
    orch.sandbox_adapter = AsyncMock()
    orch.sandbox_adapter.execute = AsyncMock(
        return_value=SimpleNamespace(
            tool_name="shell.run",
            arguments={"command": "ls"},
            stdout="ok",
            stderr="",
            exit_code=0,
            duration_ms=1,
            sandbox_id="sbx",
            execution_status=ExecutionStatus.SUCCESS,
        )
    )
    orch.db = _FakeSession()
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


def _tool_request_model(orch):
    """Return the ToolRequestModel the orchestrator created (with a real id)."""
    from nexora.backend.app.models import ToolRequest as ToolRequestModel

    for obj in orch.db.added:
        if isinstance(obj, ToolRequestModel):
            return obj
    raise AssertionError("no ToolRequestModel was added to the session")


def _approval_message_args():
    return {"message": "chore: demo commit"}


# ---------------------------------------------------------------------------
# 1. Policy: approval precedence and terminal decisions
# ---------------------------------------------------------------------------

class TestApprovalRulePrecedence:

    def test_git_commit_approved_is_allowed(self):
        """Regression: the explicit git.commit REQUIRE_APPROVAL rule deadlocked.

        Step 4 (explicit rules) returned REQUIRE_APPROVAL before the approval
        state was ever consulted, so a valid approval could never unlock a
        git.commit. A verified "approved" state must now produce ALLOW.
        """
        engine = PolicyEngine()
        d = engine.evaluate(
            Capability.GIT_COMMIT,
            _approval_message_args(),
            approval_state="approved",
        )
        assert d.decision == PolicyDecisionEnum.ALLOW
        assert "APPROVED" in d.reason_codes

    @pytest.mark.parametrize("state", [None, "pending", "rejected", "expired", "consumed"])
    def test_git_commit_not_approved_still_requires_approval(self, state):
        engine = PolicyEngine()
        d = engine.evaluate(
            Capability.GIT_COMMIT, _approval_message_args(), approval_state=state
        )
        assert d.decision == PolicyDecisionEnum.REQUIRE_APPROVAL

    def test_deny_rule_cannot_be_overridden_by_approval(self):
        engine = PolicyEngine()
        d = engine.evaluate(
            Capability.NETWORK_FETCH,
            {"url": "https://example.com"},
            approval_state="approved",
        )
        assert d.decision == PolicyDecisionEnum.DENY

    def test_git_push_deny_cannot_be_overridden_by_approval(self):
        engine = PolicyEngine()
        d = engine.evaluate(
            Capability.GIT_PUSH, {"branch": "main"}, approval_state="approved"
        )
        assert d.decision == PolicyDecisionEnum.DENY

    def test_quarantine_cannot_be_overridden_by_approval(self):
        engine = PolicyEngine()
        engine.config.rules = [
            PolicyRule(
                capability=Capability.REPO_READ,
                decision=PolicyDecisionEnum.QUARANTINE,
                reason_codes=["TEST_QUARANTINE"],
            )
        ]
        d = engine.evaluate(
            Capability.REPO_READ, {"path": "a.py"}, target_path="a.py",
            approval_state="approved",
        )
        assert d.decision == PolicyDecisionEnum.QUARANTINE

    def test_shell_run_approved_still_allows(self):
        engine = PolicyEngine()
        d = engine.evaluate(
            Capability.SHELL_RUN, {"command": "ls tests"}, approval_state="approved"
        )
        assert d.decision == PolicyDecisionEnum.ALLOW

    def test_unknown_approval_state_still_fails_closed(self):
        engine = PolicyEngine()
        d = engine.evaluate(
            Capability.GIT_COMMIT, _approval_message_args(), approval_state="forged"
        )
        assert d.decision == PolicyDecisionEnum.DENY


# ---------------------------------------------------------------------------
# 2. Approval manager lifecycle
# ---------------------------------------------------------------------------

def _new_approval(mgr: ApprovalManager, run_id=RUN_ID, tool_request_id="tr-1",
                  scope_fingerprint="fp-1", expires_in_seconds=300):
    return mgr.request_approval(
        ApprovalRequest(
            run_id=run_id,
            policy_decision_id=str(uuid.uuid4()),
            scope_description="test scope",
            expires_in_seconds=expires_in_seconds,
            tool_request_id=tool_request_id,
            scope_fingerprint=scope_fingerprint,
        )
    )


class TestApprovalLifecycle:

    def test_approval_bound_to_tool_request_is_found(self):
        """Regression: the lookup key is the tool request id, not the policy id.

        Before the fix the orchestrator looked approvals up by the tool-request
        id while they were stored against the policy-decision id, so a granted
        approval was never found.
        """
        mgr = ApprovalManager()
        rec = _new_approval(mgr, tool_request_id="tr-42")
        mgr.decide(rec.id, True, "tester", expected_run_id=RUN_ID)

        assert mgr.get_approval_state_for_tool_request("tr-42", RUN_ID) == "approved"
        # The old lookup key (policy-decision id) must not be what unlocks it.
        assert mgr.get_approval_state("tr-42") is None
        assert mgr.get_approval_state(rec.policy_decision_id) == "approved"

    def test_decide_requires_a_non_empty_approver(self):
        mgr = ApprovalManager()
        rec = _new_approval(mgr)
        with pytest.raises(ApprovalError):
            mgr.decide(rec.id, True, "")

    def test_decide_rejects_wrong_run(self):
        mgr = ApprovalManager()
        rec = _new_approval(mgr, run_id=RUN_ID)
        with pytest.raises(ApprovalError):
            mgr.decide(rec.id, True, "tester", expected_run_id=OTHER_RUN_ID)

    def test_decide_twice_is_rejected(self):
        mgr = ApprovalManager()
        rec = _new_approval(mgr)
        mgr.decide(rec.id, True, "tester")
        with pytest.raises(ApprovalError):
            mgr.decide(rec.id, False, "tester")

    def test_expired_approval_cannot_be_decided_or_used(self):
        mgr = ApprovalManager()
        rec = _new_approval(mgr, expires_in_seconds=-1)
        with pytest.raises(ApprovalError):
            mgr.decide(rec.id, True, "tester")
        assert rec.status == ApprovalStatus.EXPIRED

    def test_approved_approval_is_single_use(self):
        mgr = ApprovalManager()
        rec = _new_approval(mgr, tool_request_id="tr-1", scope_fingerprint="fp-1")
        mgr.decide(rec.id, True, "tester")

        assert mgr.consume_for_tool_request("tr-1", RUN_ID, "fp-1") is True
        assert mgr.is_approved(rec.id) is False
        assert mgr.get_approval_state_for_tool_request("tr-1", RUN_ID) == "consumed"
        # Replay is refused.
        assert mgr.consume_for_tool_request("tr-1", RUN_ID, "fp-1") is False

    def test_argument_change_invalidates_approval(self):
        mgr = ApprovalManager()
        rec = _new_approval(mgr, tool_request_id="tr-1", scope_fingerprint="fp-args-a")
        mgr.decide(rec.id, True, "tester")
        # Same tool request, different arguments -> fingerprint mismatch.
        assert mgr.consume_for_tool_request("tr-1", RUN_ID, "fp-args-b") is False
        # Still valid for the exact arguments it was granted for.
        assert mgr.consume_for_tool_request("tr-1", RUN_ID, "fp-args-a") is True

    def test_wrong_run_cannot_consume(self):
        mgr = ApprovalManager()
        rec = _new_approval(mgr, run_id=RUN_ID, tool_request_id="tr-1")
        mgr.decide(rec.id, True, "tester")
        assert mgr.consume_for_tool_request("tr-1", OTHER_RUN_ID, None) is False

    def test_rejected_approval_is_never_usable(self):
        mgr = ApprovalManager()
        rec = _new_approval(mgr, tool_request_id="tr-1")
        mgr.decide(rec.id, False, "tester")
        assert mgr.get_approval_state_for_tool_request("tr-1", RUN_ID) == "rejected"
        assert mgr.consume_for_tool_request("tr-1", RUN_ID, None) is False

    def test_no_matching_record_is_consumable_noop(self):
        mgr = ApprovalManager()
        assert mgr.consume_for_tool_request("absent", RUN_ID, "fp") is True

    def test_fingerprint_helper_is_stable_and_argument_sensitive(self):
        a = fingerprint_scope("git.commit", {"message": "x"})
        b = fingerprint_scope("git.commit", {"message": "x"})
        c = fingerprint_scope("git.commit", {"message": "y"})
        assert a == b
        assert a != c


# ---------------------------------------------------------------------------
# 3. Orchestrator integration: gating, binding, consumption
# ---------------------------------------------------------------------------

class TestOrchestratorApprovalIntegration:

    @pytest.mark.asyncio
    async def test_require_approval_pauses_and_binds_to_tool_request(self):
        orch = _make_orchestrator()
        await orch._process_tool_call(
            {"id": "c1", "name": "git.commit", "arguments": _approval_message_args()},
            None,
        )
        # Paused, never executed.
        assert orch.run.status == RunStatus.AWAITING_APPROVAL
        orch.sandbox_adapter.execute.assert_not_awaited()

        # Bound to the created tool request id and the argument fingerprint.
        tool_request_model = _tool_request_model(orch)
        pending = orch.approval_manager.get_pending_for_run(RUN_ID)
        assert len(pending) == 1
        assert pending[0].tool_request_id == str(tool_request_model.id)
        assert pending[0].scope_fingerprint == fingerprint_scope(
            "git.commit", _approval_message_args()
        )

    @pytest.mark.asyncio
    async def test_granted_approval_unlocks_exactly_that_request(self):
        orch = _make_orchestrator()
        args = _approval_message_args()
        await orch._process_tool_call(
            {"id": "c1", "name": "git.commit", "arguments": args}, None
        )
        tool_request_model = _tool_request_model(orch)
        tool_request = orch.tool_router.validate_and_route(
            "git.commit", args, "c1",
            run_id=RUN_ID, repository_id="/ws", workspace_path="/ws",
        )

        # Before approval: still gated.
        d_before = orch._authorize_tool_request(tool_request, args, tool_request_model)
        assert d_before.decision == PolicyDecisionEnum.REQUIRE_APPROVAL

        rec = orch.approval_manager.get_pending_for_run(RUN_ID)[0]
        orch.approval_manager.decide(rec.id, True, "tester", expected_run_id=RUN_ID)

        # After approval: the *same* tool request is now authorized.
        d_after = orch._authorize_tool_request(tool_request, args, tool_request_model)
        assert d_after.decision == PolicyDecisionEnum.ALLOW

        # Consuming it blocks a replay.
        fp = fingerprint_scope("git.commit", args)
        assert orch.approval_manager.consume_for_tool_request(
            str(tool_request_model.id), RUN_ID, fp
        ) is True
        d_replay = orch._authorize_tool_request(tool_request, args, tool_request_model)
        assert d_replay.decision == PolicyDecisionEnum.REQUIRE_APPROVAL

    @pytest.mark.asyncio
    async def test_deny_never_reaches_executor(self):
        orch = _make_orchestrator()
        await orch._process_tool_call(
            {"id": "c1", "name": "network.fetch", "arguments": {"url": "https://x.test"}},
            None,
        )
        orch.sandbox_adapter.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_quarantine_never_reaches_executor(self):
        from nexora.backend.app.core.policy import PolicyEngine as _PE
        eng = _PE()
        eng.config.rules = [
            PolicyRule(
                capability=Capability.REPO_READ,
                decision=PolicyDecisionEnum.QUARANTINE,
                reason_codes=["TEST"],
            )
        ]
        orch = _make_orchestrator(engine=eng)
        await orch._process_tool_call(
            {"id": "c1", "name": "repo.read", "arguments": {"path": "a.py"}}, None
        )
        orch.sandbox_adapter.execute.assert_not_awaited()
        orch._safe_stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_ordinary_allow_still_executes(self):
        orch = _make_orchestrator()
        await orch._process_tool_call(
            {"id": "c1", "name": "repo.read", "arguments": {"path": "a.py"}}, None
        )
        orch.sandbox_adapter.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_unknown_tool_never_reaches_executor(self):
        orch = _make_orchestrator()
        await orch._process_tool_call(
            {"id": "c1", "name": "explode.host", "arguments": {}}, None
        )
        orch.sandbox_adapter.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_approval_audit_event_is_correlated(self):
        orch = _make_orchestrator()
        await orch._process_tool_call(
            {"id": "c1", "name": "git.commit", "arguments": _approval_message_args()},
            None,
        )
        from nexora.backend.app.core.audit import AuditEventType

        appended = [c.args[0] for c in orch.audit_log.append.call_args_list]
        assert AuditEventType.APPROVAL_REQUESTED in appended


# ---------------------------------------------------------------------------
# 3b. Run-detail exposure: stored tool executions are redacted and bounded
# ---------------------------------------------------------------------------

def _execution_rows(orch):
    from nexora.backend.app.models import ToolExecution

    return [o for o in orch.db.added if isinstance(o, ToolExecution)]


class TestRunDetailExposure:

    @pytest.mark.asyncio
    async def test_tool_execution_arguments_are_redacted_on_store(self):
        orch = _make_orchestrator()
        orch.sandbox_adapter.execute = AsyncMock(
            return_value=SimpleNamespace(
                tool_name="repo.read",
                arguments={"path": "a.py", "authorization": "Bearer super-secret"},
                stdout="ok",
                stderr="",
                exit_code=0,
                duration_ms=1,
                sandbox_id="sbx",
                execution_status=ExecutionStatus.SUCCESS,
            )
        )
        await orch._process_tool_call(
            {"id": "c1", "name": "repo.read", "arguments": {"path": "a.py"}}, None
        )
        rows = _execution_rows(orch)
        assert len(rows) == 1
        assert rows[0].arguments["authorization"] == "[REDACTED]"
        assert rows[0].arguments["path"] == "a.py"

    @pytest.mark.asyncio
    async def test_tool_execution_output_is_bounded_without_monitoring(self):
        orch = _make_orchestrator()
        big = "x" * 200_000
        orch.sandbox_adapter.execute = AsyncMock(
            return_value=SimpleNamespace(
                tool_name="repo.read",
                arguments={"path": "a.py"},
                stdout=big,
                stderr=big,
                exit_code=0,
                duration_ms=1,
                sandbox_id="sbx",
                execution_status=ExecutionStatus.SUCCESS,
            )
        )
        await orch._process_tool_call(
            {"id": "c1", "name": "repo.read", "arguments": {"path": "a.py"}}, None
        )
        rows = _execution_rows(orch)
        assert len(rows) == 1
        assert len(rows[0].stdout) < len(big)
        assert "truncated" in rows[0].stdout
        assert len(rows[0].stderr) < len(big)
        assert "truncated" in rows[0].stderr


# ---------------------------------------------------------------------------
# 4. API route: identity, ownership, and state transitions
# ---------------------------------------------------------------------------

class _RouteDB:
    def __init__(self, run=None, approval_row=None):
        self._run = run
        self._approval_row = approval_row
        self.added = []
        self.committed = False

    async def get(self, model, ident):
        from nexora.backend.app.models import Approval, Run

        if model is Run:
            return self._run
        if model is Approval:
            return self._approval_row
        return None

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        self.committed = True


def _run_stub(run_id=RUN_ID):
    return SimpleNamespace(id=run_id, status=RunStatus.AWAITING_APPROVAL)


def _patch_manager(monkeypatch, mgr):
    import nexora.backend.app.api.routes as routes

    monkeypatch.setattr(routes, "get_approval_manager", lambda: mgr)


class TestApproveRoute:

    @pytest.mark.asyncio
    async def test_approve_uses_server_identity_and_records_event(self, monkeypatch):
        from nexora.backend.app.api.routes import approve_action
        from nexora.backend.app.models import RunEvent
        from nexora.backend.config.settings import get_settings

        mgr = ApprovalManager()
        rec = _new_approval(mgr, tool_request_id="tr-1")
        _patch_manager(monkeypatch, mgr)

        db = _RouteDB(run=_run_stub(), approval_row=SimpleNamespace())
        result = await approve_action(
            uuid.UUID(RUN_ID),
            ApproveActionRequest(approval_id=uuid.UUID(rec.id), approve=True),
            db,
        )

        assert result.status == ApprovalStatus.APPROVED
        assert result.approver_id == get_settings().OPERATOR_ID
        assert result.approver_id != "user"
        assert db._approval_row.approver_id == get_settings().OPERATOR_ID
        events = [o for o in db.added if isinstance(o, RunEvent)]
        assert len(events) == 1
        assert events[0].event_type == "approval_granted"
        assert events[0].event_metadata["approval_id"] == rec.id

    @pytest.mark.asyncio
    async def test_reject_records_rejection(self, monkeypatch):
        from nexora.backend.app.api.routes import approve_action

        mgr = ApprovalManager()
        rec = _new_approval(mgr, tool_request_id="tr-1")
        _patch_manager(monkeypatch, mgr)

        db = _RouteDB(run=_run_stub(), approval_row=SimpleNamespace())
        result = await approve_action(
            uuid.UUID(RUN_ID),
            ApproveActionRequest(approval_id=uuid.UUID(rec.id), approve=False),
            db,
        )
        assert result.status == ApprovalStatus.REJECTED

    @pytest.mark.asyncio
    async def test_approve_missing_run_is_404(self, monkeypatch):
        from nexora.backend.app.api.routes import approve_action

        mgr = ApprovalManager()
        rec = _new_approval(mgr)
        _patch_manager(monkeypatch, mgr)

        with pytest.raises(HTTPException) as exc:
            await approve_action(
                uuid.UUID(RUN_ID),
                ApproveActionRequest(approval_id=uuid.UUID(rec.id), approve=True),
                _RouteDB(run=None),
            )
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_approve_missing_approval_is_404(self, monkeypatch):
        from nexora.backend.app.api.routes import approve_action

        _patch_manager(monkeypatch, ApprovalManager())
        with pytest.raises(HTTPException) as exc:
            await approve_action(
                uuid.UUID(RUN_ID),
                ApproveActionRequest(approval_id=uuid.uuid4(), approve=True),
                _RouteDB(run=_run_stub()),
            )
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_approve_other_run_is_403(self, monkeypatch):
        from nexora.backend.app.api.routes import approve_action

        mgr = ApprovalManager()
        rec = _new_approval(mgr, run_id=OTHER_RUN_ID)
        _patch_manager(monkeypatch, mgr)

        with pytest.raises(HTTPException) as exc:
            await approve_action(
                uuid.UUID(RUN_ID),
                ApproveActionRequest(approval_id=uuid.UUID(rec.id), approve=True),
                _RouteDB(run=_run_stub()),
            )
        assert exc.value.status_code == 403

    @pytest.mark.asyncio
    async def test_approve_expired_is_409(self, monkeypatch):
        from nexora.backend.app.api.routes import approve_action

        mgr = ApprovalManager()
        rec = _new_approval(mgr, expires_in_seconds=-1)
        _patch_manager(monkeypatch, mgr)

        with pytest.raises(HTTPException) as exc:
            await approve_action(
                uuid.UUID(RUN_ID),
                ApproveActionRequest(approval_id=uuid.UUID(rec.id), approve=True),
                _RouteDB(run=_run_stub()),
            )
        assert exc.value.status_code == 409

    @pytest.mark.asyncio
    async def test_approve_twice_is_409(self, monkeypatch):
        from nexora.backend.app.api.routes import approve_action

        mgr = ApprovalManager()
        rec = _new_approval(mgr)
        _patch_manager(monkeypatch, mgr)

        await approve_action(
            uuid.UUID(RUN_ID),
            ApproveActionRequest(approval_id=uuid.UUID(rec.id), approve=True),
            _RouteDB(run=_run_stub(), approval_row=SimpleNamespace()),
        )
        with pytest.raises(HTTPException) as exc:
            await approve_action(
                uuid.UUID(RUN_ID),
                ApproveActionRequest(approval_id=uuid.UUID(rec.id), approve=False),
                _RouteDB(run=_run_stub(), approval_row=SimpleNamespace()),
            )
        assert exc.value.status_code == 409

    def test_route_source_has_no_hardcoded_approver(self):
        import nexora.backend.app.api.routes as routes

        src = inspect.getsource(routes.approve_action)
        assert '"user"' not in src
        assert "'user'" not in src
