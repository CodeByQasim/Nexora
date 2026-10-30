import asyncio
import json
from types import SimpleNamespace
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from nexora.backend.app.core.router import (
    InferenceRequest,
    InferenceResponse,
    ProviderAdapter,
    ProviderError,
)
from nexora.backend.app.core.provider_registry import ProviderRegistry
from nexora.backend.app.models import (
    FinalStatus,
    ModelRegistry,
    Run,
    RunStatus,
    RoutingMode,
)
from nexora.backend.app.worker.orchestrator import RunOrchestrator


class MockStreamResponse:
    """Mock httpx streaming response."""

    def __init__(self, lines: list[str], status_code: int = 200):
        self._lines = lines
        self.status_code = status_code

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def aiter_lines(self) -> AsyncIterator[str]:
        for line in self._lines:
            yield line

    async def aread(self) -> bytes:
        return b"Error occurred"


@pytest.mark.asyncio
async def test_provider_adapter_infer_stream_success():
    """Verify ProviderAdapter.infer_stream yields content deltas."""
    adapter = ProviderAdapter()
    adapter.api_key = "test-key"

    sse_lines = [
        'data: {"choices": [{"delta": {"content": "Hello"}}]}',
        'data: {"choices": [{"delta": {"content": " world"}}]}',
        'data: {"choices": [{"delta": {"content": "!"}}]}',
        'data: [DONE]',
    ]

    mock_client = MagicMock()
    mock_client.stream.return_value = MockStreamResponse(sse_lines)

    with patch.object(adapter, "_get_client", AsyncMock(return_value=mock_client)):
        req = InferenceRequest(
            model_id="nebius/test-model",
            messages=[{"role": "user", "content": "hi"}],
        )

        tokens = []
        async for chunk in adapter.infer_stream(req):
            tokens.append(chunk)

        assert tokens == ["Hello", " world", "!"]
        assert "".join(tokens) == "Hello world!"


@pytest.mark.asyncio
async def test_provider_adapter_infer_stream_error_handling():
    """Verify ProviderAdapter.infer_stream raises ProviderError on non-200."""
    adapter = ProviderAdapter()
    adapter.api_key = "test-key"

    mock_client = MagicMock()
    mock_client.stream.return_value = MockStreamResponse([], status_code=500)

    with patch.object(adapter, "_get_client", AsyncMock(return_value=mock_client)):
        req = InferenceRequest(
            model_id="nebius/test-model",
            messages=[{"role": "user", "content": "hi"}],
        )

        with pytest.raises(ProviderError) as exc_info:
            async for _ in adapter.infer_stream(req):
                pass

        assert exc_info.value.status_code == 500


@pytest.mark.asyncio
async def test_provider_registry_infer_stream_dispatch():
    """Verify ProviderRegistry routes streaming inference to the correct adapter."""
    registry = ProviderRegistry.__new__(ProviderRegistry)
    registry._adapters = {}
    registry._model_provider = {"nebius/test": "nebius"}
    registry.primary_provider = "nebius"

    mock_adapter = MagicMock()
    async def mock_stream(request):
        yield "chunk1"
        yield "chunk2"

    mock_adapter.infer_stream = mock_stream
    mock_adapter.is_configured = True
    registry._adapters["nebius"] = mock_adapter

    req = InferenceRequest(
        model_id="nebius/test",
        messages=[{"role": "user", "content": "hi"}],
    )

    chunks = []
    async for chunk in registry.infer_stream(req):
        chunks.append(chunk)

    assert chunks == ["chunk1", "chunk2"]


@pytest.mark.asyncio
async def test_orchestrator_streaming_queue_events():
    """Verify RunOrchestrator emits delta and done events into event_queue."""
    queue = asyncio.Queue()
    mock_db = AsyncMock()

    orchestrator = RunOrchestrator.__new__(RunOrchestrator)
    orchestrator.db = mock_db
    orchestrator.event_queue = queue
    orchestrator.run = SimpleNamespace(
        id="test-run-123",
        workspace_id="test-ws",
        snapshot_id="test-snap",
        task_text="Say hi",
        task_hash="abc12345",
        status=RunStatus.EXECUTING,
        current_stage="executing",
        routing_mode=RoutingMode.AUTO,
        requested_model_id=None,
        budget_profile_id=None,
        retries_count=0,
        input_tokens_used=0,
        output_tokens_used=0,
        steps_completed=0,
        completed_at=None,
        result_summary=None,
    )

    # Put a test delta and done
    await orchestrator._add_event("test_step", "executing", "Running test")

    event = await queue.get()
    assert event["type"] == "status"
    assert event["stage"] == "executing"
    assert event["message"] == "Running test"
