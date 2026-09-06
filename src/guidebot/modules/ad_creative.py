"""Advertising creative workflow built on the provider-neutral model interface."""

from __future__ import annotations

from typing import Any

from guidebot.events import EventBus
from guidebot.model_serving import ModelBackend, ModelRequest
from guidebot.scheduler import Task


class AdCreativeModule:
    """Generate a reviewable brief; publishing is deliberately outside this module."""

    name = "ad_creative"

    def __init__(self, backend: ModelBackend) -> None:
        self.backend = backend
        self.event_bus: EventBus | None = None
        self.running = False

    def start(self, event_bus: EventBus) -> None:
        self.event_bus = event_bus
        self.running = True

    def stop(self) -> None:
        self.running = False

    def handle_task(self, task: Task) -> dict[str, Any]:
        response = self.backend.infer(
            ModelRequest(
                task="ad_creative_brief",
                modality="multimodal",
                payload={
                    "product": task.payload.get("product", "Guidebot"),
                    "audience": task.payload.get("audience", "科技爱好者"),
                    "goal": task.payload.get("goal", "提升点击率"),
                    "prompt": task.payload.get("text", "生成一份广告素材方案"),
                },
                metadata={"task_id": task.task_id, "skill_id": task.skill_id},
            )
        )
        return {
            "module": self.name,
            "action": task.action,
            "creative_brief": response.text,
            "image_url": response.image_url,
            "backend": response.metadata.get("backend", self.backend.name),
            "latency_ms": response.latency_ms,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "tokens_per_second": response.tokens_per_second,
            "requires_human_review": True,
            "published": False,
        }
