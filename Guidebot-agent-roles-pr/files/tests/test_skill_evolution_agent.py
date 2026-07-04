from __future__ import annotations

from guidebot.agents import SkillEvolutionAgent
from guidebot.models import Action, ActionKind, Decision, Event, RobotState, Trajectory
from guidebot.observation import Observation
from guidebot.policy_evolution import PolicyEvolution
from guidebot.reflection import EnvironmentFeedback
from guidebot.self_evolving import build_default_library


def test_skill_evolution_agent_turns_failure_into_candidate_skill() -> None:
    library = build_default_library()
    agent = SkillEvolutionAgent(
        library,
        evolution=PolicyEvolution(failure_threshold=1),
    )
    observation = Observation(22, 50, 0, 1, "user.message")
    action = Action(ActionKind.SPEAK, {"text": "我不确定。"}, "bad conversation response")
    trajectory = Trajectory(
        Event("user.message", "我有点不舒服"),
        Decision((action,), "我不确定。", "conversation fallback"),
        (action,),
        (),
    )
    feedback = EnvironmentFeedback(
        -1.0,
        False,
        observation,
        "misread user comfort intent",
        {"failure_mode": "user_signal_misread"},
    )

    report = agent.observe(
        observation=observation,
        trajectory=trajectory,
        state=RobotState(),
        skill_name="conversation",
        feedback=feedback,
    )

    assert report.memory_size == 1
    assert report.attribution.should_evolve_skill
    assert report.critique.failure_mode == "user_signal_misread"
    assert report.outcome.triggered
    assert report.outcome.added
    assert report.outcome.generated_skill is not None
    assert report.outcome.generated_skill.generated
