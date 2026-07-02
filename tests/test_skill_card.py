from __future__ import annotations

from guidebot.models import Decision, RobotState
from guidebot.observation import Observation
from guidebot.skills import Skill, SkillLibrary


def _skill(name: str) -> Skill:
    def precondition(observation: Observation, state: RobotState) -> bool:
        return True

    def policy(observation: Observation, state: RobotState) -> Decision:
        return Decision()

    return Skill(name, policy, precondition, ("observe",), (0, 0, 0, 0))


def test_old_skill_gets_v1_card_and_child_lineage_is_preserved() -> None:
    library = SkillLibrary((_skill("base"),))
    parent = library.card("base")
    child_skill = _skill("base_recovery")
    candidate = library.create_candidate_card(child_skill, parent_name="base")

    library.activate(child_skill, candidate.with_validation(0.2, accepted=True))
    lineage = library.lineage("base_recovery")

    assert parent.version == "v1"
    assert lineage[1].version == "v2"
    assert lineage[1].parent_id == parent.skill_id
    assert tuple(card.skill_id for card in lineage) == (parent.skill_id, candidate.skill_id)


def test_skill_card_tracks_success_and_known_failures() -> None:
    library = SkillLibrary((_skill("tracked"),))
    library.record_outcome("tracked", success=True)
    card = library.record_outcome("tracked", success=False, failure_mode="delay")

    assert card.success_count == 1
    assert card.failure_count == 1
    assert card.known_failure_modes == ("delay",)

