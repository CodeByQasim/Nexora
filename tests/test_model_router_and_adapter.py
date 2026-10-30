import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from packages.contracts.schemas import ModelRequest, ModelResponse
from packages.model_router import ModelRouter
from packages.nebius_adapter.mock import MockModelProvider


def test_mock_provider_success() -> None:
    """Verify MockModelProvider returns valid text responses and usage metrics."""
    provider = MockModelProvider()
    request = ModelRequest(prompt="Hello, world!", task_complexity="low")
    response = provider.complete(request)

    assert isinstance(response, ModelResponse)
    assert response.content is not None
    assert "Mock reply" in response.content
    assert response.input_tokens > 0
    assert response.output_tokens > 0
    assert response.total_tokens == response.input_tokens + response.output_tokens
    assert response.latency_ms >= 0
    assert response.model_used
    assert response.routing_reason


def test_task_complexity_rejects_unsupported_tier() -> None:
    with pytest.raises(ValidationError):
        ModelRequest(prompt="Hello", task_complexity="complex")


@pytest.mark.parametrize("token_budget", [0, -1])
def test_model_request_rejects_non_positive_token_budget(token_budget: int) -> None:
    with pytest.raises(ValidationError):
        ModelRequest(prompt="Hello", token_budget=token_budget)


def test_nebius_adapter_passes_token_budget() -> None:
    from packages.nebius_adapter.nebius import NebiusAdapter

    adapter = NebiusAdapter.__new__(NebiusAdapter)
    captured: dict[str, Any] = {}

    class Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs: Any) -> Any:
                    captured.update(kwargs)
                    return SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(content="ok", tool_calls=None),
                                finish_reason="stop",
                            )
                        ],
                        usage=SimpleNamespace(
                            prompt_tokens=1,
                            completion_tokens=1,
                            total_tokens=2,
                        ),
                    )

    adapter.client = Client()  # type: ignore[assignment]
    request = ModelRequest(prompt="test", token_budget=123)

    adapter.complete(request, model_id="nebius/test-model")

    assert captured["max_tokens"] == 123


def test_package_bridge_refresh_keeps_stale_models_available(monkeypatch: Any) -> None:
    from nexora.backend.app.core.package_bridge import (
        NEBIUS_MODELS_CACHE_TTL_SECONDS,
        PackageBridge,
    )
    from packages.nebius_adapter.base import ModelProvider

    started = Event()
    release = Event()

    class SlowAdapter:
        calls = 0

        def fetch_available_models(self) -> list[str]:
            self.calls += 1
            started.set()
            assert release.wait(timeout=5)
            return ["fresh-model"]

    monkeypatch.setattr(ModelProvider, "api_key", "test-key")
    monkeypatch.setattr(ModelProvider, "base_url", "https://example.invalid")
    bridge = PackageBridge()
    adapter = SlowAdapter()
    bridge._nebius = adapter  # type: ignore[assignment]
    bridge._nebius_models = ["cached-model"]
    bridge._nebius_models_fetched_at = (
        time.monotonic() - NEBIUS_MODELS_CACHE_TTL_SECONDS - 1
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        refresh = executor.submit(bridge.providers)
        assert started.wait(timeout=1)
        try:
            concurrent_result = executor.submit(bridge.providers).result(timeout=1)
            assert concurrent_result[0]["available_models"] == ["cached-model"]
        finally:
            release.set()

        refreshed_result = refresh.result(timeout=1)

    assert refreshed_result[0]["available_models"] == ["fresh-model"]
    assert adapter.calls == 1


def test_tool_call_parsing() -> None:
    """Verify model tool calls are correctly parsed into ModelResponse.tool_calls."""
    provider = MockModelProvider()
    request = ModelRequest(
        prompt="Call a tool",
        tools=[{"type": "function", "function": {"name": "mock_tool", "arguments": "{}"}}],
        task_complexity="moderate",
    )
    response = provider.complete(request)

    assert response.tool_calls is not None
    assert len(response.tool_calls) > 0
    assert "id" in response.tool_calls[0]
    assert response.finish_reason == "tool_calls"


def test_malformed_response() -> None:
    """Verify graceful error handling when API/mock returns missing structures."""
    from packages.nebius_adapter.nebius import NebiusAdapter
    from packages.contracts.schemas import ModelRequest

    adapter = NebiusAdapter.__new__(NebiusAdapter)

    class BadClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs: Any) -> Any:
                    class BadCompletion:
                        pass

                    return BadCompletion()

    adapter.client = BadClient()  # type: ignore[assignment]
    adapter._router = ModelRouter()  # type: ignore[assignment]
    adapter._redact = lambda text: text  # type: ignore[method-assign]

    request = ModelRequest(prompt="test", task_complexity="low")

    with pytest.raises((AttributeError, IndexError, Exception)):
        adapter.complete(request)


def test_timeout_and_provider_error() -> None:
    """Verify timeout errors and API failures raise predictable exceptions."""
    from packages.nebius_adapter.nebius import NebiusAdapter
    from packages.contracts.schemas import ModelRequest

    adapter = NebiusAdapter.__new__(NebiusAdapter)

    class FailingClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs: Any) -> Any:
                    raise RuntimeError("Connection timeout")

    adapter.client = FailingClient()  # type: ignore[assignment]
    adapter._router = ModelRouter()  # type: ignore[assignment]
    adapter._redact = lambda text: "Connection timeout"  # type: ignore[method-assign]

    request = ModelRequest(prompt="test", task_complexity="low")

    with pytest.raises(RuntimeError) as exc_info:
        adapter.complete(request)

    assert "Connection timeout" in str(exc_info.value)


def test_secret_redaction() -> None:
    """Verify API keys are redacted to [REDACTED_API_KEY] in errors."""
    from packages.nebius_adapter.nebius import NebiusAdapter

    adapter = NebiusAdapter.__new__(NebiusAdapter)
    adapter.api_key = "test-secret-key-123"

    error_message = "Error calling API with key test-secret-key-123"
    redacted = adapter._redact(error_message)

    assert "test-secret-key-123" not in redacted
    assert "[REDACTED]" in redacted


def test_model_router_tier_selection() -> None:
    """Test routing for low/moderate/high tiers including manual overrides."""
    router = ModelRouter()
    available_models = []

    req_low = ModelRequest(prompt="hi", task_complexity="low")
    model, reason = router.select_model(req_low, available_models)
    assert model == "meta-llama/Meta-Llama-3.1-8B-Instruct"
    assert "low" in reason

    req_mod = ModelRequest(prompt="hi", task_complexity="moderate")
    model, reason = router.select_model(req_mod, available_models)
    assert model == "meta-llama/Meta-Llama-3.1-70B-Instruct"
    assert "moderate" in reason

    req_high = ModelRequest(prompt="hi", task_complexity="high")
    model, reason = router.select_model(req_high, available_models)
    assert model == "deepseek-ai/DeepSeek-R1-0528"
    assert "high" in reason

    req_override = ModelRequest(
        prompt="hi",
        task_complexity="high",
        preferred_model="nebius/custom-model",
    )
    model, reason = router.select_model(req_override, ["nebius/custom-model"])
    assert model == "nebius/custom-model"
    assert "preferred_model_override" in reason

    req_override_unavail = ModelRequest(
        prompt="hi",
        task_complexity="low",
        preferred_model="nebius/unavailable",
    )
    model, reason = router.select_model(req_override_unavail, ["other/model"])
    assert model == "nebius/unavailable"
    assert "unavailable" in reason


def test_package_model_router_uses_latest_registry_rows() -> None:
    from nexora.backend.app.core.router import get_model_router

    original = SimpleNamespace(model_id="old/model", is_enabled=True)
    updated = SimpleNamespace(model_id="new/model", is_enabled=True)

    assert "old/model" in get_model_router([original]).enabled_models
    refreshed_router = get_model_router([updated])

    assert "new/model" in refreshed_router.enabled_models
    assert "old/model" not in refreshed_router.enabled_models


@pytest.mark.asyncio
async def test_nebius_registry_rejects_missing_api_key() -> None:
    from nexora.backend.app.core.provider_registry import ProviderRegistry
    from nexora.backend.app.core.router import InferenceRequest, ProviderAdapter, ProviderError

    adapter = ProviderAdapter()
    adapter.api_key = ""
    registry = ProviderRegistry.__new__(ProviderRegistry)
    registry.primary_provider = "nebius"
    registry._adapters = {"nebius": adapter}
    registry._model_provider = {"test-model": "nebius"}

    with pytest.raises(ProviderError, match="not configured"):
        await registry.infer(InferenceRequest(model_id="test-model", messages=[]))


@pytest.mark.asyncio
async def test_nebius_adapter_preserves_versioned_base_path() -> None:
    import httpx
    from nexora.backend.app.core.router import InferenceRequest, ProviderAdapter

    requested_urls = []

    def respond(request: httpx.Request) -> httpx.Response:
        requested_urls.append(request.url)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}]},
        )

    adapter = ProviderAdapter()
    adapter.api_key = "test-key"
    adapter._client = httpx.AsyncClient(
        base_url=adapter.base_url,
        transport=httpx.MockTransport(respond),
    )
    try:
        await adapter.infer(InferenceRequest(model_id="test-model", messages=[]))
    finally:
        await adapter.close()

    assert requested_urls[0].path == "/v1/chat/completions"


@pytest.mark.asyncio
async def test_plain_model_response_is_saved_for_chat() -> None:
    from nexora.backend.app.models import FinalStatus
    from nexora.backend.app.worker.orchestrator import RunOrchestrator

    orchestrator = RunOrchestrator.__new__(RunOrchestrator)
    orchestrator.run = SimpleNamespace(result_summary=None)
    orchestrator.current_diff = None
    orchestrator._complete_run = AsyncMock()
    orchestrator._run_validation = AsyncMock()

    await orchestrator._handle_model_response("Hello! How can I help?")

    assert orchestrator.run.result_summary == {
        "type": "assistant_response",
        "content": "Hello! How can I help?",
    }
    orchestrator._complete_run.assert_awaited_once_with(FinalStatus.COMPLETED)
    orchestrator._run_validation.assert_not_awaited()


@pytest.mark.asyncio
async def test_text_mentioning_patch_is_not_misclassified_as_a_diff() -> None:
    from nexora.backend.app.models import FinalStatus
    from nexora.backend.app.worker.orchestrator import RunOrchestrator

    orchestrator = RunOrchestrator.__new__(RunOrchestrator)
    orchestrator.run = SimpleNamespace(result_summary=None)
    orchestrator.current_diff = None
    orchestrator._complete_run = AsyncMock()
    orchestrator._run_validation = AsyncMock()

    await orchestrator._handle_model_response("I can explain how to create a patch.")

    assert orchestrator.run.result_summary["content"] == "I can explain how to create a patch."
    orchestrator._complete_run.assert_awaited_once_with(FinalStatus.COMPLETED)
    orchestrator._run_validation.assert_not_awaited()


@pytest.mark.asyncio
async def test_unified_diff_model_response_enters_validation() -> None:
    from nexora.backend.app.worker.orchestrator import RunOrchestrator

    patch = "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-old\n+new"
    orchestrator = RunOrchestrator.__new__(RunOrchestrator)
    orchestrator.run = SimpleNamespace(result_summary=None)
    orchestrator.current_diff = None
    orchestrator._complete_run = AsyncMock()
    orchestrator._run_validation = AsyncMock()

    await orchestrator._handle_model_response(patch)

    assert orchestrator.current_diff == patch
    orchestrator._run_validation.assert_awaited_once()
    orchestrator._complete_run.assert_not_awaited()


def test_run_prompt_intent_separates_chat_from_code_tasks() -> None:
    from nexora.backend.app.worker.orchestrator import _is_code_task_prompt

    assert not _is_code_task_prompt("Reply with exactly: Nemotron connection OK")
    assert not _is_code_task_prompt("What is Python used for?")
    assert _is_code_task_prompt("Write Java code for an if/else statement")
    assert _is_code_task_prompt("Create a Python hello world program")
    assert _is_code_task_prompt("Fix the empty-input bug in the calculator")


def test_repository_context_is_only_requested_for_existing_code_changes() -> None:
    from nexora.backend.app.worker.orchestrator import _needs_repository_context

    assert not _needs_repository_context("Write a Python function add(a, b)")
    assert _needs_repository_context("Fix the existing calculator bug")


def test_internal_reasoning_preamble_is_rejected() -> None:
    from nexora.backend.app.worker.orchestrator import _is_internal_reasoning

    assert _is_internal_reasoning("Here's a thinking process: ...")
    assert _is_internal_reasoning("Analysis: ...")
    assert not _is_internal_reasoning("Hello! How can I help?")


def test_demo_calculator_request_builds_local_write_tool_call() -> None:
    from nexora.backend.app.worker.orchestrator import (
        _local_demo_calculator_tool_call,
    )

    tool_call = _local_demo_calculator_tool_call("Create a basic calculator")

    assert tool_call is not None
    assert tool_call["name"] == "repo.write"
    assert tool_call["arguments"]["path"] == "calculator.py"
    compile(tool_call["arguments"]["content"], "calculator.py", "exec")
    assert _local_demo_calculator_tool_call("Create a basic website") is None


def test_validator_checks_already_applied_workspace_patch(tmp_path: Any) -> None:
    from nexora.backend.app.core.validator import ValidatorCritic
    from nexora.backend.app.models import ValidationStatus

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=tmp_path,
        check=True,
    )
    source = tmp_path / "app.py"
    source.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "app.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "baseline"], cwd=tmp_path, check=True)

    source.write_text("value = 2\n", encoding="utf-8")
    patch = subprocess.run(
        ["git", "diff", "--relative", "--", "app.py"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    validator = ValidatorCritic(str(tmp_path))

    assert validator._check_patch_applies(patch).status == ValidationStatus.FAIL
    assert (
        validator._check_patch_applies(patch, already_applied=True).status
        == ValidationStatus.PASS
    )
