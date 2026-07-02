"""Batch simulation runner for skills and skill libraries."""

from __future__ import annotations

from statistics import mean

from guidebot.models import Action, ActionKind, Reading, RobotState, SensorKind
from guidebot.router import HierarchicalRouter
from guidebot.safety import SafetyPolicy
from guidebot.skills import Skill, SkillLibrary

from .evaluator import (
    EvaluationMetrics,
    EvaluationReport,
    ScenarioResult,
    comfort_error,
    evolution_accept_rate,
    recovery_steps,
    skill_reuse_rate,
)
from .room_dynamics import RoomDynamics
from .scenarios import Scenario, default_stress_scenarios


class SimulationSuite:
    def __init__(
        self,
        scenarios: tuple[Scenario, ...] | None = None,
        *,
        steps: int = 8,
        safety: SafetyPolicy | None = None,
        router: HierarchicalRouter | None = None,
    ) -> None:
        if steps < 1:
            raise ValueError("steps must be positive")
        self.scenarios = scenarios or default_stress_scenarios()
        self.steps = steps
        self.safety = safety or SafetyPolicy()
        self.router = router or HierarchicalRouter()

    def run(
        self,
        candidate: Skill | SkillLibrary,
        *,
        evolution_accepted: int = 0,
        evolution_proposed: int = 0,
    ) -> EvaluationReport:
        results = tuple(self._run_scenario(scenario, candidate) for scenario in self.scenarios)
        all_temperatures = tuple(value for result in results for value in result.temperatures)
        all_skills = tuple(name for result in results for name in result.selected_skills)
        preferred = mean(
            float(scenario.expected_behavior.get("preferred_temperature_c", 23.0))
            for scenario in self.scenarios
        )
        metrics = EvaluationMetrics(
            comfort_error(all_temperatures, preferred),
            mean(result.recovered_in_steps for result in results),
            sum(result.unsafe_actions for result in results),
            sum(result.safety_rejections for result in results),
            skill_reuse_rate(all_skills),
            evolution_accept_rate(evolution_accepted, evolution_proposed),
        )
        return EvaluationReport(mean(result.score for result in results), metrics, results)

    def _run_scenario(
        self,
        scenario: Scenario,
        candidate: Skill | SkillLibrary,
    ) -> ScenarioResult:
        dynamics = RoomDynamics(scenario.initial_observation, scenario.environment_params)
        observation = scenario.initial_observation
        state = RobotState()
        temperatures: list[float] = []
        selected_skills: list[str] = []
        unsafe = 0
        rejections = 0
        action_kinds: list[str] = []
        preferred = float(scenario.expected_behavior.get("preferred_temperature_c", 23.0))
        tolerance = float(scenario.expected_behavior.get("comfort_tolerance_c", 1.0))

        for step_index in range(self.steps):
            scheduled = scenario.user_feedback_schedule.get(step_index)
            if isinstance(scheduled, (int, float)):
                preferred = float(scheduled)
            elif isinstance(scheduled, dict) and "preferred_temperature_c" in scheduled:
                preferred = float(scheduled["preferred_temperature_c"])

            state.update(Reading(SensorKind.TEMPERATURE, observation.temperature_c, "°C", "simulation"))
            state.update(Reading(SensorKind.HUMIDITY, observation.humidity_pct, "%", "simulation"))
            skill = self._select(candidate, observation, state)
            selected_skills.append(skill.name)
            decision = skill.execute(observation, state)
            accepted_hvac: Action | None = None
            for action in decision.actions:
                action_kinds.append(action.kind.value)
                safety_result = self.safety.evaluate(action, state)
                if not safety_result.allowed:
                    rejections += 1
                    unsafe += 1
                elif action.kind is ActionKind.SET_HVAC:
                    accepted_hvac = action
            room_step = dynamics.step(accepted_hvac)
            observation = room_step.observation
            temperatures.append(room_step.true_temperature_c)

        error = comfort_error(temperatures, preferred)
        recovered = recovery_steps(temperatures, preferred, tolerance)
        required_action = scenario.expected_behavior.get("required_action")
        required_met = required_action is None or str(required_action) in action_kinds
        met = error <= float(scenario.expected_behavior.get("max_comfort_error", 5.0)) and required_met
        score = 1.0 / (1.0 + error) + 0.1 * float(required_met) - 2.0 * unsafe
        return ScenarioResult(
            scenario.name,
            score,
            tuple(temperatures),
            tuple(selected_skills),
            unsafe,
            rejections,
            recovered,
            met and unsafe == 0,
        )

    def _select(
        self,
        candidate: Skill | SkillLibrary,
        observation,
        state: RobotState,
    ) -> Skill:
        if isinstance(candidate, Skill):
            return candidate
        return self.router.route(observation, state, candidate).skill

