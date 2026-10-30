"""
Offline mock provider -- no network calls, for tests and local development.
"""
import time

from packages.contracts.schemas import ModelRequest, ModelResponse
from packages.nebius_adapter.base import ModelProvider

MOCK_MODEL_ID = "mock/nemotron-super"

MOCK_MODELS = [
    "mock/nemotron-nano",
    "mock/nemotron-super",
    "mock/nemotron-ultra",
]


class MockModelProvider(ModelProvider):
    """Drop-in :class:`ModelProvider` that fabricates responses locally."""

    def fetch_available_models(self) -> list[str]:
        """Return the hardcoded mock model list."""
        return list(MOCK_MODELS)

    def complete(
        self,
        request: ModelRequest,
        model_id: str = MOCK_MODEL_ID,
        routing_reason: str = "mock",
    ) -> ModelResponse:
        """Return a canned :class:`ModelResponse` without touching the network."""
        started = time.perf_counter()

        tool_calls = None
        if request.tools:
            tool_calls = [
                {
                    "id": "mock-call-0",
                    "type": "function",
                    "function": {"name": "mock_tool", "arguments": "{}"},
                }
            ]
        content = None if tool_calls else f"Mock reply to: {request.prompt}"

        input_tokens = max(1, len(request.prompt) // 4)
        output_tokens = max(1, len(content or "mock") // 4)

        return ModelResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else "stop",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            model_used=model_id,
            routing_reason=routing_reason,
        )
