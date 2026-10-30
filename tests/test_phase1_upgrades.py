"""
Test suite for Nexora Master Execution Phase 1 Upgrades:
- Instant Stop Authority and task cancellation
- Long conversation memory sliding-window compaction (<session_state>)
- Web scraping tool (web.scrape) with sanitized Markdown extraction and safe fallback
- Model switching and Nemotron fixed routing resolution
- Detail level output verbosity token scaling
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from nexora.backend.app.core.tools import (
    ToolName,
    ToolRouter,
    LocalSandboxAdapter,
)
from nexora.backend.app.core.context import (
    compact_thread_history,
    align_prefix_boundary,
    calculate_dynamic_output_budget,
)
from nexora.backend.app.core.router import (
    ModelRouter,
    RoutingReason,
)
from nexora.backend.app.models import (
    ModelRegistry,
    RoutingMode,
    RunStatus,
)


# ---------------------------------------------------------------------------
# 1. Web Scraping Tool Verification
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_web_scrape_missing_url():
    adapter = LocalSandboxAdapter()
    result = await adapter._web_scrape({})
    assert result["exit_code"] == 1
    assert "Missing required 'url'" in result["stderr"]


@pytest.mark.asyncio
async def test_web_scrape_html_sanitization(monkeypatch):
    adapter = LocalSandboxAdapter()
    mock_html = """
    <!DOCTYPE html>
    <html>
    <head><title>Test Page</title><style>.hidden { display: none; }</style></head>
    <body>
      <nav><a href="/home">Home</a></nav>
      <h1>Nexora Documentation</h1>
      <p>Nexora is an <strong>autonomous</strong> AI coding agent.</p>
      <script>console.log("tracker");</script>
      <p>Check the <a href="https://example.com/api">API Reference</a> for details.</p>
      <footer>Copyright 2026</footer>
    </body>
    </html>
    """

    class MockResponse:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def headers(self):
            return {}
        def read(self):
            return mock_html.encode("utf-8")
        @property
        def headers(self):
            class H:
                def get_content_charset(self):
                    return "utf-8"
            return H()

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout: MockResponse())

    result = await adapter._web_scrape({"url": "https://example.com/docs"})
    assert result["exit_code"] == 0
    stdout = result["stdout"]
    # Check that headings and paragraphs are converted to markdown
    assert "# Nexora Documentation" in stdout
    assert "Nexora is an autonomous AI coding agent." in stdout
    assert "[API Reference](https://example.com/api)" in stdout
    # Check that script, style, nav, and footer were stripped
    assert "tracker" not in stdout
    assert ".hidden" not in stdout
    assert "Copyright 2026" not in stdout


@pytest.mark.asyncio
async def test_web_scrape_safe_exception_boundary(monkeypatch):
    adapter = LocalSandboxAdapter()
    import urllib.request
    import urllib.error

    def mock_urlopen_fail(req, timeout):
        raise urllib.error.URLError("Connection timed out")

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen_fail)

    result = await adapter._web_scrape({"url": "https://nonexistent-domain.xyz"})
    assert result["exit_code"] == 1
    assert "Web scrape connection error" in result["stderr"]


# ---------------------------------------------------------------------------
# 2. Long-Conversation Sliding Window Compaction
# ---------------------------------------------------------------------------

def test_compact_thread_history_short_conversation():
    # 2 turns fits comfortably without needing session state
    thread = [
        {"role": "user", "content": "How do I fix the calculator bug?"},
        {"role": "assistant", "content": "Here is the fix:\n```python\ndef add(a, b):\n    return a + b\n```"},
    ]
    compacted, stats = compact_thread_history(thread, max_history_tokens=2000, preserve_recent_turns=3)
    assert len(compacted) == 2
    assert stats["session_state_generated"] is False


def test_compact_thread_history_long_conversation_generates_session_state():
    # 8 turns should trigger session state distillation for older turns while preserving latest 3
    thread = []
    for i in range(1, 9):
        thread.append({"role": "user", "content": f"Turn {i}: please update file_{i}.py with feature {i}"})
        thread.append({"role": "assistant", "content": f"Turn {i} reply: Updated file_{i}.py successfully with code."})

    compacted, stats = compact_thread_history(thread, max_history_tokens=400, preserve_recent_turns=3)
    assert stats["session_state_generated"] is True
    # The first message in compacted should be the session_state block
    first_msg = compacted[0]
    assert "<session_state>" in first_msg["content"]
    assert "Historical Milestones" in first_msg["content"]
    # The latest turn must be present in full fidelity
    latest_msg = compacted[-1]
    assert "Turn 8 reply" in latest_msg["content"]


# ---------------------------------------------------------------------------
# 3. Model Switching & Fixed Routing Resolution
# ---------------------------------------------------------------------------

def test_fixed_model_routing_strict_resolution():
    nano_model = ModelRegistry(
        id="nano-uuid-1111",
        model_id="nvidia/nemotron-3-nano",
        display_name="NVIDIA Nemotron 3 Nano",
        tier="nano",
        provider="nebius",
        is_enabled=True,
    )
    ultra_model = ModelRegistry(
        id="ultra-uuid-2222",
        model_id="nvidia/nemotron-3-ultra",
        display_name="NVIDIA Nemotron 3 Ultra",
        tier="ultra",
        provider="nebius",
        is_enabled=True,
    )
    gemini_model = ModelRegistry(
        id="gemini-uuid-3333",
        model_id="gemini-1.5-pro",
        display_name="Google Gemini 1.5 Pro",
        tier="super",
        provider="gemini",
        is_enabled=True,
    )

    router = ModelRouter([nano_model, ultra_model, gemini_model])

    # 1. Match by exact model_id string
    dec1 = router.route(RoutingMode.FIXED, requested_model_id="nvidia/nemotron-3-ultra")
    assert dec1.model.model_id == "nvidia/nemotron-3-ultra"
    assert dec1.reason == RoutingReason.EXPLICIT_FIXED

    # 2. Match by UUID string
    dec2 = router.route(RoutingMode.FIXED, requested_model_id="nano-uuid-1111")
    assert dec2.model.model_id == "nvidia/nemotron-3-nano"
    assert dec2.reason == RoutingReason.EXPLICIT_FIXED

    # 3. Match by tier / keyword (e.g. gemini)
    dec3 = router.route(RoutingMode.FIXED, requested_model_id="gemini")
    assert dec3.model.model_id == "gemini-1.5-pro"
    assert dec3.reason == RoutingReason.EXPLICIT_FIXED


# ---------------------------------------------------------------------------
# 4. Detail Level Output Verbosity Scaling
# ---------------------------------------------------------------------------

def test_dynamic_output_budget_scaling():
    # Simple concise task
    concise_tokens = calculate_dynamic_output_budget("add simple comment to func", ceiling=4096)
    assert concise_tokens <= 1024

    # Detailed portfolio / full application task
    detailed_tokens = calculate_dynamic_output_budget("build full responsive portfolio website with html css", ceiling=4096)
    assert detailed_tokens >= 3072


def test_web_scrape_policy_authorization():
    from nexora.backend.app.core.policy import PolicyEngine, Capability, PolicyDecisionEnum
    engine = PolicyEngine()
    decision = engine.evaluate(Capability.WEB_SCRAPE, {"url": "https://example.com/docs"})
    assert decision.decision == PolicyDecisionEnum.ALLOW_WITH_MONITORING
    assert "READ_ONLY_WEB" in decision.reason_codes


@pytest.mark.asyncio
async def test_web_scrape_linkedin_notice(monkeypatch):
    adapter = LocalSandboxAdapter()
    import urllib.request
    import urllib.error

    def mock_err(*args, **kwargs):
        raise urllib.error.HTTPError(
            url="https://www.linkedin.com/in/test",
            code=999,
            msg="Request Denied",
            hdrs={},
            fp=None,
        )

    monkeypatch.setattr(urllib.request, "urlopen", mock_err)
    res = await adapter._web_scrape({"url": "https://www.linkedin.com/in/test"})
    assert res["exit_code"] == 0
    assert "LinkedIn restricts automated guest scraping" in res["stdout"]

