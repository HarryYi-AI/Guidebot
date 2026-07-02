"""Dynamically extensible deterministic skill library L_t."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

from .models import Decision, RobotState
from .observation import Observation
from .skill_card import SkillCard

SkillPolicy = Callable[[Observation, RobotState], Decision]
Precondition = Callable[[Observation, RobotState], bool]


@dataclass(frozen=True, slots=True)
class Skill:
    """A reusable ``f_skill: O → A`` with explicit applicability and effects."""

    name: str
    policy: SkillPolicy
    precondition: Precondition
    effects: tuple[str, ...]
    route_weights: tuple[float, ...]
    level: int = 0
    generated: bool = False

    def applicable(self, observation: Observation, state: RobotState) -> bool:
        return self.precondition(observation, state)

    def execute(self, observation: Observation, state: RobotState) -> Decision:
        return self.policy(observation, state)


class SkillLibrary:
    """Stores ``L = {(name, f, precondition, effect)}`` and supports ``L ∪ {f_new}``."""

    def __init__(self, skills: tuple[Skill, ...] = ()) -> None:
        self._skills: dict[str, Skill] = {}
        self._cards_by_id: dict[str, SkillCard] = {}
        self._active_cards: dict[str, SkillCard] = {}
        for skill in skills:
            self.add(skill)

    def add(
        self,
        skill: Skill,
        *,
        replace: bool = False,
        card: SkillCard | None = None,
    ) -> None:
        if skill.name in self._skills and not replace:
            raise ValueError(f"skill already exists: {skill.name}")
        metadata = card or SkillCard.create(
            skill.name,
            description=f"Executable skill: {skill.name}",
            preconditions=("callable precondition",),
            action_schema={"effects": skill.effects},
            safety_scope=skill.effects,
            accepted=True,
        )
        if metadata.name != skill.name:
            raise ValueError("skill and SkillCard names must match")
        if not metadata.accepted:
            metadata = metadata.with_validation(metadata.validation_score, accepted=True)
        self._skills[skill.name] = skill
        self._cards_by_id[metadata.skill_id] = metadata
        self._active_cards[skill.name] = metadata

    def create_candidate_card(
        self,
        skill: Skill,
        *,
        parent_name: str | None = None,
        description: str = "",
    ) -> SkillCard:
        parent = self._active_cards.get(parent_name) if parent_name else None
        if parent is not None:
            child = parent.child(name=skill.name, description=description or None)
            return SkillCard(
                child.skill_id,
                child.name,
                child.version,
                child.parent_id,
                child.description,
                child.preconditions,
                {"effects": skill.effects},
                skill.effects,
                accepted=False,
            )
        return SkillCard.create(
            skill.name,
            description=description,
            preconditions=("callable precondition",),
            action_schema={"effects": skill.effects},
            safety_scope=skill.effects,
        )

    def activate(self, skill: Skill, card: SkillCard) -> SkillCard:
        accepted_card = card if card.accepted else card.with_validation(card.validation_score, accepted=True)
        self.add(skill, replace=skill.name in self._skills, card=accepted_card)
        return accepted_card

    def get(self, name: str) -> Skill:
        try:
            return self._skills[name]
        except KeyError as error:
            raise KeyError(f"unknown skill: {name}") from error

    def find(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def card(self, name_or_id: str) -> SkillCard:
        card = self._active_cards.get(name_or_id) or self._cards_by_id.get(name_or_id)
        if card is None:
            raise KeyError(f"unknown skill card: {name_or_id}")
        return card

    def lineage(self, name_or_id: str) -> tuple[SkillCard, ...]:
        current = self.card(name_or_id)
        lineage = [current]
        while current.parent_id is not None:
            current = self._cards_by_id[current.parent_id]
            lineage.append(current)
        return tuple(reversed(lineage))

    def record_outcome(
        self,
        name: str,
        *,
        success: bool,
        failure_mode: str | None = None,
    ) -> SkillCard:
        updated = self.card(name).record_outcome(success=success, failure_mode=failure_mode)
        self._active_cards[name] = updated
        self._cards_by_id[updated.skill_id] = updated
        return updated

    def __iter__(self) -> Iterator[Skill]:
        return iter(self._skills.values())

    def __len__(self) -> int:
        return len(self._skills)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._skills)
