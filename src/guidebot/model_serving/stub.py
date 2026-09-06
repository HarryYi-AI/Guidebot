"""Deterministic backend for offline demos, tests, and CI."""

from __future__ import annotations

from time import perf_counter

from .models import ModelRequest, ModelResponse


class DeterministicBackend:
    name = "deterministic"

    def infer(self, request: ModelRequest) -> ModelResponse:
        started = perf_counter()
        product = str(request.payload.get("product", "产品"))
        audience = str(request.payload.get("audience", "目标用户"))
        goal = str(request.payload.get("goal", "提升转化"))
        if request.task == "ad_creative_brief":
            text = (
                f"主标题：让{product}更懂{audience}；"
                f"画面：产品主体居中，使用场景作背景；"
                f"CTA：立即了解；目标：{goal}。"
            )
        else:
            text = f"deterministic response for {request.task}"
        latency_ms = (perf_counter() - started) * 1_000
        return ModelResponse(
            text=text,
            latency_ms=latency_ms,
            output_tokens=max(1, len(text) // 2),
            metadata={"backend": self.name, "offline": True},
        )
