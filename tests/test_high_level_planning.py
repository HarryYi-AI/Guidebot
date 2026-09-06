import pytest

from guidebot.models import ActionKind, RobotState
from guidebot.planning import FixedOptionCompiler, HighLevelPlan, OptionStep, SkillOption


def test_wait_option_produces_no_low_level_action() -> None:
    decision = FixedOptionCompiler().compile(
        HighLevelPlan((OptionStep(SkillOption.WAIT),), rationale="insufficient evidence"),
        RobotState(),
    )

    assert decision.actions == ()


def test_alarm_option_compiles_to_typed_action() -> None:
    decision = FixedOptionCompiler().compile(
        HighLevelPlan((OptionStep(SkillOption.ALARM, {"time": "07:00"}),)),
        RobotState(),
    )

    assert decision.actions[0].kind is ActionKind.SET_ALARM
    assert decision.actions[0].parameters["time"] == "07:00"


def test_plan_length_is_bounded() -> None:
    plan = HighLevelPlan(tuple(OptionStep(SkillOption.WAIT) for _ in range(3)))

    with pytest.raises(ValueError, match="exceeds"):
        FixedOptionCompiler(max_steps=2).compile(plan, RobotState())
