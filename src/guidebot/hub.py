"""Central coordinator for perception, decisions, safety, and action."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

from .agent import Agent
from .bus import EventBus
from .devices.base import DeviceAdapter
from .models import ActionKind, Event, Reading, RobotState, Trajectory
from .safety import SafetyPolicy
from .self_evolving import SelfEvolvingAgent
from .reflection import EnvironmentFeedback

FeedbackProvider = Callable[[Trajectory, RobotState], EnvironmentFeedback]


class GuidebotHub:
    def __init__(
        self,
        device: DeviceAdapter,
        agent: Agent | None = None,
        safety: SafetyPolicy | None = None,
        bus: EventBus | None = None,
        feedback_provider: FeedbackProvider | None = None,
    ) -> None:
        self.device = device
        self.agent = agent or SelfEvolvingAgent()
        self.safety = safety or SafetyPolicy()
        self.bus = bus or EventBus()
        self.feedback_provider = feedback_provider
        self.state = RobotState()
        self.trajectories: list[Trajectory] = []

    async def start(self) -> None:
        await self.device.start()
        self.state.health = "ready"
        await self.bus.publish(Event("system.ready", {"device": self.device.name}))

    async def stop(self) -> None:
        await self.device.stop()
        self.state.health = "stopped"
        await self.bus.publish(Event("system.stopped", {}))

    async def ingest(self, reading: Reading) -> Trajectory:
        self.state.update(reading)
        return await self._handle(Event("sensor.reading", reading))

    async def say(self, text: str) -> Trajectory:
        return await self._handle(Event("user.message", text))

    async def run_once(self) -> Trajectory | None:
        """Read and process one hardware sample; useful for loops and integration tests."""
        reading = await self.device.read()
        return await self.ingest(reading) if reading is not None else None

    def record_feedback(self, index: int, score: float, feedback: str = "") -> Trajectory:
        """Attach verifier/user feedback without mutating the original audit record."""
        if not 0 <= score <= 1:
            raise ValueError("score must be within 0..1")
        updated = replace(self.trajectories[index], score=score, feedback=feedback or None)
        self.trajectories[index] = updated
        return updated

    async def _handle(self, event: Event) -> Trajectory:
        await self.bus.publish(event)
        decision = await self.agent.decide(event, self.state)
        accepted = []
        rejected = []
        for action in decision.actions:
            result = self.safety.evaluate(action, self.state)
            if result.allowed:
                await self.device.execute(action)
                accepted.append(action)
                if action.kind is ActionKind.SET_HVAC:
                    self.state.hvac_target_c = float(action.parameters["target_c"])
                await self.bus.publish(Event("action.executed", action))
            else:
                rejected.append(action)
                await self.bus.publish(Event("action.rejected", {"action": action, "reason": result.reason}))

        trajectory = Trajectory(event, decision, tuple(accepted), tuple(rejected))
        self.trajectories.append(trajectory)
        outcome_learner = getattr(self.agent, "observe_outcome", None)
        if outcome_learner is not None:
            feedback = (
                self.feedback_provider(trajectory, self.state) if self.feedback_provider else None
            )
            outcome_learner(trajectory, self.state, feedback)
        return trajectory
