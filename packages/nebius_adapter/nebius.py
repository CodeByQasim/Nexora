"""
Nebius Token Factory adapter -- OpenAI-compatible chat completions.
"""
import logging
import time
from typing import Any

from openai import OpenAI

from packages.contracts.schemas import ModelRequest, ModelResponse
from packages.model_router.router import ModelRouter
from packages.nebius_adapter.base import ModelProvider

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
REDACTED = "[REDACTED]"


class NebiusAdapter(ModelProvider):
    """Sends chat completion requests to the Nebius Token Factory API."""

    def __init__(self) -> None:
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )
        self._router = ModelRouter()

    def _redact(self, text: str) -> str:
        """Strip the API key from any string leaving this adapter."""
        if not self.api_key:
            return text
        return text.replace(self.api_key, REDACTED)

    def fetch_available_models(self) -> list[str]:
        """Return model IDs advertised by the provider, or ``[]`` on failure."""
        try:
            return [model.id for model in self.client.models.list().data]
        except Exception as exc:
            logger.warning("Nebius model listing failed: %s", self._redact(str(exc)))
            return []

    def complete(
        self,
        request: ModelRequest,
        model_id: str = DEFAULT_MODEL_ID,
        routing_reason: str = "default",
    ) -> ModelResponse:
        """Run one chat completion and map it onto a :class:`ModelResponse`.

        API errors are re-raised with ``self.api_key`` removed from the
        message so no secret reaches logs, telemetry or the caller.
        """
        # Integrate with ModelRouter when no explicit model_id is specified
        # If the default model_id is used (not explicitly overridden), use router
        selected_model_id = model_id
        selected_routing_reason = routing_reason

        # Check if this is using the default (meaning no explicit override)
        if model_id == DEFAULT_MODEL_ID and routing_reason == "default":
            available_models = self.fetch_available_models()
            selected_model_id, selected_routing_reason = self._router.select_model(
                request, available_models
            )

        kwargs: dict[str, Any] = {
            "model": selected_model_id,
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": request.token_budget,
        }
        if request.tools:
            kwargs["tools"] = request.tools

        started = time.perf_counter()
        try:
            completion = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            message = self._redact(str(exc))
            logger.error("Nebius completion failed: %s", message)
            raise RuntimeError(message) from None
        latency_ms = (time.perf_counter() - started) * 1000.0

        choice = completion.choices[0]
        tool_calls = None
        if choice.message.tool_calls:
            tool_calls = [call.model_dump() for call in choice.message.tool_calls]

        usage = completion.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else input_tokens + output_tokens

        return ModelResponse(
            content=choice.message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            model_used=selected_model_id,
            routing_reason=selected_routing_reason,
        )
