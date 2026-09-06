"""Provider-neutral request and response contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """One inference request independent of DashScope, vLLM, or Diffusers."""

    task: str
    payload: dict[str, Any]
    modality: str = "text"
    timeout_s: float = 30.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelResponse:
    text: str | None = None
    image_url: str | None = None
    latency_ms: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def tokens_per_second(self) -> float | None:
        if not self.output_tokens or self.latency_ms <= 0:
            return None
        return self.output_tokens / (self.latency_ms / 1_000)
