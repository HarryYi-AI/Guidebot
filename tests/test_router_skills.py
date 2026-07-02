from __future__ import annotations

import pytest

from guidebot.models import Decision, RobotState
from guidebot.observation import FeatureMapper, Observation
from guidebot.router import HierarchicalRouter
from guidebot.skills import Skill, SkillLibrary


def _decision(observation: Observation, state: RobotState) -> Decision:
    return Decision(rationale="test")


def _always(observation: Observation, state: RobotState) -> bool:
    return True


def test_feature_map_and_sigmoid_router_match_equations() -> None:
    observation = Observation(32, 50, 0, 0)
    mapper = FeatureMapper()
    high = Skill("high", _decision, _always, (), (2, 0, 0, 0))
    low = Skill("low", _decision, _always, (), (-2, 0, 0, 0))
    library = SkillLibrary((high, low))

    route = HierarchicalRouter(mapper).route(observation, RobotState(), library)

    assert mapper(observation) == (1.0, 0.0, 0.0, 0.0)
    assert route.skill.name == "high"
    assert route.score == pytest.approx(0.880797, rel=1e-5)


def test_hierarchical_router_prioritizes_earlier_level() -> None:
    observation = Observation(22, 50, 0, 0)
    urgent = Skill("urgent", _decision, _always, (), (0, 0, 0, 0), level=0)
    social = Skill("social", _decision, _always, (), (10, 0, 0, 0), level=2)

    route = HierarchicalRouter().route(
        observation,
        RobotState(),
        SkillLibrary((social, urgent)),
    )

    assert route.skill.name == "urgent"


def test_skill_library_add_get_iterate_and_reject_duplicate() -> None:
    skill = Skill("one", _decision, _always, ("effect",), (0, 0, 0, 0))
    library = SkillLibrary()
    library.add(skill)

    assert library.get("one") is skill
    assert tuple(library) == (skill,)
    assert library.names == ("one",)
    with pytest.raises(ValueError, match="already exists"):
        library.add(skill)

