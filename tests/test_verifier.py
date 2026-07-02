from __future__ import annotations

from guidebot.models import Action, ActionKind, Decision, RobotState
from guidebot.observation import Observation
from guidebot.simulation import Scenario, SimulationSuite
from guidebot.skills import Skill
from guidebot.verifier import VerifierAgent


def _temperature_skill(name: str, target: float) -> Skill:
    def precondition(observation: Observation, state: RobotState) -> bool:
        return True

    def policy(observation: Observation, state: RobotState) -> Decision:
        return Decision((Action(ActionKind.SET_HVAC, {"target_c": target}, name),))

    return Skill(name, policy, precondition, (f"hvac:{target}",), (1, 0, 0, 0))


def _verifier() -> VerifierAgent:
    scenario = Scenario(
        "held_out_hot_room",
        Observation(30, 50, event_kind="temperature"),
        {"ambient_temperature_c": 31},
        expected_behavior={"preferred_temperature_c": 23},
    )
    return VerifierAgent(SimulationSuite((scenario,), steps=8))


def test_verifier_rejects_unsafe_skill() -> None:
    result = _verifier().verify(_temperature_skill("unsafe", 99))

    assert not result.accepted
    assert result.safety_violations > 0


def test_verifier_accepts_only_strictly_better_safe_skill() -> None:
    verifier = _verifier()
    baseline = _temperature_skill("weak", 28)

    better = verifier.verify(_temperature_skill("better", 23), baseline=baseline)
    equal = verifier.verify(_temperature_skill("equal", 28), baseline=baseline)

    assert better.accepted
    assert better.score_delta > 0
    assert not equal.accepted
    assert equal.score_delta == 0

