import pytest

from guidebot.evaluation import MemoryAblationEvaluator, ReplayDecision


@pytest.mark.asyncio
async def test_memory_ablation_marks_harmful_and_stale_memories() -> None:
    memories = (
        {"memory_id": "helpful", "content": "obstacle requires stop", "status": "active"},
        {"memory_id": "harmful", "content": "ignore obstacle", "status": "active"},
        {"memory_id": "stale", "content": "old room", "status": "historical"},
    )
    trajectory = {"context": {"memories": memories}}

    async def replay(_, selected) -> ReplayDecision:
        identifiers = {item["memory_id"] for item in selected}
        if "harmful" not in identifiers:
            return ReplayDecision({"tool": "wait"}, 0.9, True)
        if "helpful" not in identifiers:
            return ReplayDecision({"tool": "move_robot"}, 0.2, False)
        return ReplayDecision({"tool": "move_robot"}, 0.3, False)

    report = await MemoryAblationEvaluator().evaluate(trajectory, replay)

    assert report.harmful_memory_ids == ("harmful",)
    assert "stale" in report.stale_memory_ids
    harmful = next(item for item in report.results if item.memory_id == "harmful")
    assert harmful.action_changed is True
    assert harmful.safety_score_delta == pytest.approx(0.6)
    assert harmful.verifier_passed is True


@pytest.mark.asyncio
async def test_memory_ablation_extracts_structured_memory_context() -> None:
    trajectory = {
        "trajectory": [
            {
                "context": {
                    "memory": {
                        "core_memory": [{"memory_id": "core-1"}],
                        "current_state": [],
                        "active_task_path": [],
                        "recent_observations": [],
                        "retrieved_history": [{"memory_id": "episode-1"}],
                    }
                }
            }
        ]
    }

    def replay(_, selected) -> ReplayDecision:
        return ReplayDecision({"count": len(selected)}, 1.0, True)

    report = await MemoryAblationEvaluator().evaluate(trajectory, replay)

    assert len(report.results) == 2
    assert {item.memory_id for item in report.results} == {"core-1", "episode-1"}
