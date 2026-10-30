"""Model router for selecting models based on task complexity and availability."""
from typing import List, Tuple

from packages.contracts.schemas import ModelRequest


class ModelRouter:
    """Routes requests to appropriate models based on task complexity.

    Provides tiered model selection with fallback logic when the preferred
    model is unavailable or does not match availability constraints.
    """

    MODEL_TIERS = {
        "low": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "moderate": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "high": "deepseek-ai/DeepSeek-R1-0528",
    }

    def select_model(
        self, request: ModelRequest, available_models: List[str]
    ) -> Tuple[str, str]:
        """Select an appropriate model for the given request.

        Args:
            request: Provider-neutral model request containing routing hints.
            available_models: List of model IDs available from the provider.

        Returns:
            A tuple of (selected_model_id, routing_reason).
        """
        if request.preferred_model:
            preferred = request.preferred_model
            if not available_models or preferred in available_models:
                return preferred, "preferred_model_override"
            return preferred, "preferred_model_override_unavailable"

        task_complexity = request.task_complexity or "low"

        if task_complexity == "low":
            tier_model = self.MODEL_TIERS["low"]
            reason = "low_complexity"
        elif task_complexity == "moderate":
            tier_model = self.MODEL_TIERS["moderate"]
            reason = "moderate_complexity"
        elif task_complexity == "high":
            tier_model = self.MODEL_TIERS["high"]
            reason = "high_complexity"
        else:
            tier_model = self.MODEL_TIERS["low"]
            reason = "default_low_complexity"

        if not available_models:
            return tier_model, f"{reason}_no_available_list"

        if tier_model in available_models:
            return tier_model, reason

        if available_models:
            return available_models[0], f"{reason}_fallback_to_first_available"

        return tier_model, f"{reason}_fallback"
