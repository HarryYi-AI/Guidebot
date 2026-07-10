"""Unified Guidebot runtime orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .events import Event, EventBus
from .intent import Intent
from .intent_analyzer import IntentAnalyzer
from .logbook import RuntimeLogger
from .modules import (
    AlarmTimerModule,
    ClimateControlModule,
    HealthMonitorModule,
    MobilityModule,
    SceneMonitorModule,
    VoiceChatModule,
)
from .safety import RuntimeSafetyState, SafetyGate, SafetyResult
from .scheduler import Scheduler, Task


@dataclass(frozen=True, slots=True)
class RuntimeTrace:
    event: Event
    intent: Intent
    task: Task | None
    safety: SafetyResult | None
    action: dict[str, Any] | None


class GuidebotRuntime:
    """EventBus → IntentAnalyzer → Scheduler → SafetyGate → Modules."""

    def __init__(
        self,
        *,
        logger: RuntimeLogger | None = None,
        safety_state: RuntimeSafetyState | None = None,
    ) -> None:
        self.event_bus = EventBus()
        self.analyzer = IntentAnalyzer()
        self.scheduler = Scheduler()
        self.safety = SafetyGate()
        self.safety_state = safety_state or RuntimeSafetyState()
        self.logger = logger
        self.modules = {
            "voice_chat": VoiceChatModule(),
            "scene_monitor": SceneMonitorModule(),
            "health_monitor": HealthMonitorModule(),
            "alarm_timer": AlarmTimerModule(),
            "mobility": MobilityModule(),
            "climate_control": ClimateControlModule(),
        }
        for module in self.modules.values():
            module.start(self.event_bus)

    def stop(self) -> None:
        for module in self.modules.values():
            module.stop()

    def ingest(self, event: Event) -> RuntimeTrace:
        self.event_bus.publish(event)
        if event.event_type == "ultrasonic.obstacle" and event.payload.get("obstacle") is True:
            self.safety_state.obstacle = True

        intent = self.analyzer.analyze(event)
        task = self.scheduler.schedule(intent)
        safety = None
        action = None
        if task is not None:
            safety = self.safety.evaluate_task(task, self.safety_state)
            if safety.allowed:
                action = self.execute(task)
                if task.priority >= 100 and not task.interruptible:
                    self.safety_state.active_safety_alert = True
            else:
                action = {
                    "module": task.target_module,
                    "action": task.action,
                    "blocked": True,
                    "reason": safety.reason,
                }

        trace = RuntimeTrace(event, intent, task, safety, action)
        self._log(trace)
        return trace

    def execute(self, task: Task) -> dict[str, Any]:
        module = self.modules[task.target_module]
        return module.handle_task(task)

    def _log(self, trace: RuntimeTrace) -> None:
        if self.logger is None:
            return
        self.logger.event(trace.event)
        self.logger.intent(trace.intent)
        if trace.task is not None:
            self.logger.task({"task": trace.task, "safety": trace.safety, "action": trace.action})
