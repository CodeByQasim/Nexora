"""
Shared provider-neutral contracts.

The schemas here are the only shapes that cross the boundary between the
router and any model provider adapter.
"""
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class ModelRequest(BaseModel):
    """Provider-neutral completion request."""

    prompt: str = Field(..., description="User/system prompt to complete")
    tools: Optional[list[dict[str, Any]]] = Field(
        default=None, description="Tool definitions the model may call"
    )
    task_complexity: Literal["low", "moderate", "high"] = Field(
        default="low", description="Routing hint: low, moderate or high"
    )
    token_budget: int = Field(
        default=1000, gt=0, description="Maximum output tokens"
    )
    preferred_model: Optional[str] = Field(
        default=None, description="Model to use when the caller pins one"
    )


class ModelResponse(BaseModel):
    """Provider-neutral completion response."""

    content: Optional[str] = Field(default=None, description="Assistant text, if any")
    tool_calls: Optional[list[dict[str, Any]]] = Field(
        default=None, description="Tool calls requested by the model"
    )
    finish_reason: str = Field(..., description="Why generation stopped")
    input_tokens: int = Field(..., description="Tokens consumed by the prompt")
    output_tokens: int = Field(..., description="Tokens generated")
    total_tokens: int = Field(..., description="input_tokens + output_tokens")
    latency_ms: float = Field(..., description="Round-trip latency in milliseconds")
    model_used: str = Field(..., description="Model that served the request")
    routing_reason: str = Field(..., description="Why this model was chosen")
