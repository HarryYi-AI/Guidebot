from __future__ import annotations

from guidebot.models import Action, ActionKind, Decision, RobotState
from guidebot.observation import Observation
from guidebot.simulation import Scenario, SimulationSuite
from guidebot.skills import Skill


def _cooling_skill(target: float) -> Skill:
    def precondition(observation: Observation, state: RobotState) -> bool:
        return True

    def policy(observation: Observation, state: RobotState) -> Decision:
        action = Action(ActionKind.SET_HVAC, {"target_c": target}, "simulation")
        return Decision((action,))

    return Skill(f"cool_{target}", policy, precondition, (f"hvac:{target}",), (1, 0, 0, 0))


def test_simulation_suite_reports_required_metrics() -> None:
    scenario = Scenario(
        "delayed_humid_room",
        Observation(30, 80, event_kind="temperature"),
        {"ambient_temperature_c": 31, "ac_delay_steps": 1, "humidity_factor": 0.7},
        expected_behavior={"preferred_temperature_c": 23, "comfort_tolerance_c": 1},
    )

    report = SimulationSuite((scenario,), steps=8).run(_cooling_skill(23))

    assert report.metrics.comfort_error > 0
    assert report.metrics.recovery_steps >= 1
    assert report.metrics.unsafe_action_count == 0
    assert report.metrics.safety_rejection_count == 0
    assert report.metrics.skill_reuse_rate > 0


def test_simulation_counts_unsafe_actions() -> None:
    scenario = Scenario("unsafe", Observation(30, 50, event_kind="temperature"))
    report = SimulationSuite((scenario,), steps=2).run(_cooling_skill(99))

    assert report.metrics.unsafe_action_count == 2
    assert report.metrics.safety_rejection_count == 2

