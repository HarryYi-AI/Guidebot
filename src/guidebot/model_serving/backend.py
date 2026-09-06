"""Structural interface for replaceable model providers."""

from __future__ import annotations

from typing import Protocol

from .models import ModelRequest, ModelResponse


class ModelBackend(Protocol):
    """A provider can be local vLLM, a cloud API, or a deterministic test double."""

    name: str

    def infer(self, request: ModelRequest) -> ModelResponse:
        """Run one inference request or raise a provider-specific exception."""

