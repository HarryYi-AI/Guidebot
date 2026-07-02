from __future__ import annotations

from guidebot.memory import Experience, MemoryStream
from guidebot.models import Decision
from guidebot.observation import Observation
from guidebot.policy_evolution import PolicyEvolution
from guidebot.reflection import Critique, EnvironmentFeedback
from guidebot.self_evolving import build_default_library


def _failure(mode: str = "user_signal_misread") -> Experience:
    observation = Observation(22, 50, 0, 1, "user.message")
    feedback = EnvironmentFeedback(-1, False, observation, "misread")
    critique = Critique("ambiguous intent", mode, 0.8, "ask for clarification", 0.9, 0.1)
    return Experience(observation, "conversation", Decision(), feedback, critique)


def test_policy_evolution_triggers_only_at_failure_threshold() -> None:
    memory = MemoryStream()
    library = build_default_library()
    evolution = PolicyEvolution(failure_threshold=3)
    memory.add(_failure())
    memory.add(_failure())

    before = evolution.evolve(memory, library)
    memory.add(_failure())
    after = evolution.evolve(memory, library)

    assert not before.triggered
    assert after.triggered and after.added
    assert after.generated_skill is not None
    assert after.generated_skill.generated
    assert len(library) == 7

