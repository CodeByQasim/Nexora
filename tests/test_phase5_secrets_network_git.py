"""
Phase 5 — Secrets, provider, network and Git security tests
==========================================================
Covers the migration-free Phase 5 hardening of the egress and secret surface:

* provider error text derived from an untrusted remote body is bounded,
  control-character-stripped and credential-redacted before it can reach a
  log line, an exception message or an API response;
* ``network.fetch`` has no implementation path (deny stub) and is ``DENY`` in
  policy, so a task sandbox cannot reach internal/metadata addresses;
* ``git.push`` has no implementation path (deny stub) and is ``DENY`` in policy;
* ``git.commit`` still requires approval;
* tool arguments are redacted (secrets replaced, long strings truncated) before
  they are written to logs or audit records.

No real external service is contacted: the Nebius adapter is exercised against
an in-memory ``httpx.Response`` and the tool adapters against their deny stubs.
"""
import asyncio

import httpx
import pytest

from nexora.backend.app.core.policy import Capability, PolicyEngine
from nexora.backend.app.core.router import (
    ProviderAdapter,
    ProviderError,
    _MAX_PROVIDER_ERROR_CHARS,
    _sanitize_provider_error,
)
from nexora.backend.app.core.tools import get_sandbox_adapter, redact_arguments
from nexora.backend.app.schemas import PolicyDecision

SECRET = "nb-sk-test-abcdef123456"


# ---------------------------------------------------------------------------
# 5.A / 5.B — provider error text: bounded + credential-redacted
# ---------------------------------------------------------------------------

def test_sanitize_bounds_length():
    text = _sanitize_provider_error("z" * 100_000)
    assert len(text) == _MAX_PROVIDER_ERROR_CHARS


def test_sanitize_strips_control_characters():
    text = _sanitize_provider_error("a\x00b\x07c\x1bd")
    assert "\x00" not in text
    assert "\x07" not in text
    assert "\x1b" not in text


def test_sanitize_redacts_secret():
    text = _sanitize_provider_error(f"invalid key {SECRET} presented", SECRET)
    assert SECRET not in text
    assert "[REDACTED]" in text


def test_sanitize_prevents_crlf_log_forging():
    """A hostile provider body must not forge a second log line (CR/LF/ANSI)."""
    forged = (
        "bad request\r\nINFO 2026-01-01 auth: api_key=abcd accepted\r\n"
        "\x1b[2J"
    )
    text = _sanitize_provider_error(forged, None)
    assert "\r" not in text
    assert "\n" not in text
    assert "\x1b" not in text
    # The injected markers survive only as inert text on a single line.
    assert text.count("INFO") == 1


def _adapter() -> ProviderAdapter:
    adapter = ProviderAdapter()
    adapter.api_key = SECRET
    return adapter


def test_error_text_prefers_provider_message_and_bounds_it():
    adapter = _adapter()
    response = httpx.Response(
        400, json={"error": {"message": "malformed request " + "x" * 5_000}}
    )
    text = adapter._error_text(response)
    assert len(text) <= _MAX_PROVIDER_ERROR_CHARS
    assert text.startswith("malformed request")


def test_error_text_never_leaks_api_key():
    adapter = _adapter()
    response = httpx.Response(
        401, json={"error": {"message": f"invalid api key {SECRET} provided"}}
    )
    text = adapter._error_text(response)
    assert SECRET not in text
    assert "[REDACTED]" in text


def test_error_text_bounds_non_json_body():
    adapter = _adapter()
    response = httpx.Response(500, text="<html>" + "y" * 50_000 + "</html>")
    text = adapter._error_text(response)
    assert len(text) <= _MAX_PROVIDER_ERROR_CHARS


def test_infer_client_error_is_sanitized():
    """A 4xx must surface a bounded, redacted message through ProviderError."""
    from nexora.backend.app.core.router import InferenceRequest

    adapter = _adapter()

    class _FakeClient:
        is_closed = False

        async def post(self, *args, **kwargs):
            return httpx.Response(
                400, json={"error": {"message": f"bad key {SECRET} " + "p" * 20_000}}
            )

        async def aclose(self):
            pass

    adapter._client = _FakeClient()
    request = InferenceRequest(
        model_id="test-model", messages=[{"role": "user", "content": "hi"}]
    )

    with pytest.raises(ProviderError) as excinfo:
        asyncio.run(adapter.infer(request))

    message = str(excinfo.value)
    assert SECRET not in message
    assert len(message) <= _MAX_PROVIDER_ERROR_CHARS + 40


# ---------------------------------------------------------------------------
# 5.C — network egress / SSRF
# ---------------------------------------------------------------------------

def test_network_fetch_is_denied_by_policy():
    engine = PolicyEngine()
    decision = engine.evaluate(
        Capability.NETWORK_FETCH, {"url": "http://169.254.169.254/latest/meta-data/"}
    )
    assert decision.decision == PolicyDecision.DENY


def test_network_fetch_has_no_implementation_path():
    adapter = get_sandbox_adapter()
    result = asyncio.run(adapter._network_fetch({"url": "http://127.0.0.1:6379/"}))
    assert result["exit_code"] == 1
    assert "denied" in result["stderr"].lower()


# ---------------------------------------------------------------------------
# 5.E — git / external side effects
# ---------------------------------------------------------------------------

def test_git_push_is_denied_by_policy():
    engine = PolicyEngine()
    decision = engine.evaluate(Capability.GIT_PUSH, {"remote": "origin"})
    assert decision.decision == PolicyDecision.DENY


def test_git_push_has_no_implementation_path():
    adapter = get_sandbox_adapter()
    result = asyncio.run(adapter._git_push({"remote": "origin"}, "/ws", {}))
    assert result["exit_code"] == 1
    assert "denied" in result["stderr"].lower()


def test_git_commit_still_requires_approval():
    engine = PolicyEngine()
    decision = engine.evaluate(
        Capability.GIT_COMMIT, {"path": "a.py", "message": "wip"}
    )
    assert decision.decision == PolicyDecision.REQUIRE_APPROVAL


# ---------------------------------------------------------------------------
# 5.A — argument redaction for logs/audit
# ---------------------------------------------------------------------------

def test_redact_arguments_replaces_sensitive_keys():
    arguments = {"api_key": SECRET, "db_password": "hunter2", "path": "a.py"}
    safe = redact_arguments(arguments)
    assert safe["api_key"] == "[REDACTED]"
    assert safe["db_password"] == "[REDACTED]"
    assert safe["path"] == "a.py"
    # The original mapping must not be mutated.
    assert arguments["api_key"] == SECRET


def test_redact_arguments_truncates_long_strings():
    arguments = {"content": "x" * 100_000}
    safe = redact_arguments(arguments, max_string_length=64)
    assert len(safe["content"]) < len(arguments["content"])
    assert "truncated" in safe["content"]
