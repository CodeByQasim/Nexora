"""
Provider-neutral model provider interface.

Concrete adapters (Nebius, Gemini, ...) subclass :class:`ModelProvider` and
implement the two abstract methods; the router depends only on this interface.
"""
import os
from abc import ABC, abstractmethod

from packages.contracts.schemas import ModelRequest, ModelResponse


class ModelProvider(ABC):
    """Abstract base class every model provider adapter must implement.

    Connection settings are read from the environment:

    - ``NEBIUS_BASE_URL``: provider API base URL
    - ``NEBIUS_API_KEY``: provider API key
    - ``NEBIUS_TIMEOUT``: request timeout in seconds
    """

    base_url: str = os.getenv("NEBIUS_BASE_URL", "")
    api_key: str = os.getenv("NEBIUS_API_KEY", "")
    timeout: float = float(os.getenv("NEBIUS_TIMEOUT", "120"))

    @abstractmethod
    def complete(self, request: ModelRequest) -> ModelResponse:
        """Send ``request`` to the provider and return its response."""

    @abstractmethod
    def fetch_available_models(self) -> list[str]:
        """Return the model identifiers this provider can serve."""
