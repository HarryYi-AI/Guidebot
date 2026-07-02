"""Experience-driven skill synthesis π_{t+1} = π_t ⊕ Δπ."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Protocol, Sequence

from .memory import Experience, MemoryStream
from .models import Action, ActionKind, Decision, RobotState
from .observation import FeatureMapper, Observation
from .skills import Skill, SkillLibrary
from .skill_card import SkillCard
from .verifier import VerificationResult, VerifierAgent


@dataclass(frozen=True, slots=True)
class FailureCluster:
    failure_mode: str
    experiences: tuple[Experience, ...]
    centroid: tuple[float, ...]
    dominant_event_kind: str


@dataclass(frozen=True, slots=True)
class EvolutionOutcome:
    triggered: bool
    cluster: FailureCluster | None = None
    generated_skill: Skill | None = None
    added: bool = False
    reason: str = ""
    skill_card: SkillCard | None = None
    verification: VerificationResult | None = None


@dataclass(frozen=True, slots=True)
class RejectedSkill:
    skill: Skill
    card: SkillCard
    reason: str
    verification: VerificationResult


class SkillSynthesizer(Protocol):
    def synthesize(self, cluster: FailureCluster, library: SkillLibrary) -> Skill: ...


class FailurePatternClusterer:
    """Clusters explicit failures by their structured ``failure_mode`` label."""

    def __init__(self, mapper: FeatureMapper | None = None) -> None:
        self.mapper = mapper or FeatureMapper()

    def cluster(self, failures: Sequence[Experience]) -> tuple[FailureCluster, ...]:
        groups: dict[str, list[Experience]] = defaultdict(list)
        for experience in failures:
            groups[experience.critique.failure_mode].append(experience)

        clusters = []
        for mode, items in groups.items():
            vectors = [self.mapper(item.observation) for item in items]
            centroid = tuple(sum(values) / len(values) for values in zip(*vectors))
            event_kind = Counter(item.observation.event_kind for item in items).most_common(1)[0][0]
            clusters.append(FailureCluster(mode, tuple(items), centroid, event_kind))
        return tuple(sorted(clusters, key=lambda cluster: len(cluster.experiences), reverse=True))


class RecoverySkillSynthesizer:
    """Safe baseline synthesizer; an LLM can replace it behind the same interface."""

    def synthesize(self, cluster: FailureCluster, library: SkillLibrary) -> Skill:
        slug = re.sub(r"[^a-z0-9]+", "_", cluster.failure_mode.lower()).strip("_")
        name = f"recovery_{slug or 'unknown'}"
        event_kind = cluster.dominant_event_kind
        suggestion = cluster.experiences[-1].critique.suggested_action

        def precondition(observation: Observation, state: RobotState) -> bool:
            return observation.event_kind == event_kind

        def policy(observation: Observation, state: RobotState) -> Decision:
            action = Action(
                ActionKind.NOTIFY,
                {"level": "warning", "message": suggestion},
                f"recovery for recurring failure: {cluster.failure_mode}",
            )
            return Decision((action,), suggestion, f"synthesized from {len(cluster.experiences)} failures")

        # The centroid is the linear router prototype w_k for this failure region.
        return Skill(
            name=name,
            policy=policy,
            precondition=precondition,
            effects=(f"mitigates:{cluster.failure_mode}", "notifies_user"),
            route_weights=cluster.centroid,
            level=1,
            generated=True,
        )


class PolicyEvolution:
    """Triggers only when a failure cluster reaches an experience threshold."""

    def __init__(
        self,
        failure_threshold: int = 3,
        clusterer: FailurePatternClusterer | None = None,
        synthesizer: SkillSynthesizer | None = None,
        verifier: VerifierAgent | None = None,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be positive")
        self.failure_threshold = failure_threshold
        self.clusterer = clusterer or FailurePatternClusterer()
        self.synthesizer = synthesizer or RecoverySkillSynthesizer()
        self.verifier = verifier or VerifierAgent()
        self.rejected_skill_buffer: list[RejectedSkill] = []

    def evolve(self, memory: MemoryStream, library: SkillLibrary) -> EvolutionOutcome:
        attributable_failures = tuple(
            experience
            for experience in memory.failures()
            if experience.attribution is None or experience.attribution.should_evolve_skill
        )
        clusters = self.clusterer.cluster(attributable_failures)
        eligible = next(
            (cluster for cluster in clusters if len(cluster.experiences) >= self.failure_threshold),
            None,
        )
        if eligible is None:
            return EvolutionOutcome(False, reason="failure threshold not reached")

        skill = self.synthesizer.synthesize(eligible, library)
        parent_name = eligible.experiences[-1].skill_name
        if library.find(skill.name) is not None and skill.name != parent_name:
            return EvolutionOutcome(True, eligible, skill, False, "skill already exists")
        card = library.create_candidate_card(
            skill,
            parent_name=parent_name,
            description=f"Synthesized from recurring {eligible.failure_mode} failures",
        )
        baseline = library.find(parent_name)
        verification = self.verifier.verify(skill, card=card, baseline=baseline)
        validated_card = card.with_validation(
            verification.score_delta,
            accepted=verification.accepted,
        )
        if not verification.accepted:
            self.rejected_skill_buffer.append(
                RejectedSkill(skill, validated_card, verification.reason, verification)
            )
            return EvolutionOutcome(
                True,
                eligible,
                skill,
                False,
                verification.reason,
                validated_card,
                verification,
            )
        active_card = library.activate(skill, validated_card)
        return EvolutionOutcome(
            True,
            eligible,
            skill,
            True,
            "candidate passed verifier and entered active library",
            active_card,
            verification,
        )
