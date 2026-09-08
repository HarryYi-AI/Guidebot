import json

from guidebot.memory import TaskNodeStatus, TaskStateTree


def test_task_state_tracks_active_path_and_switches_failed_branch() -> None:
    tree = TaskStateTree("完成休息提醒", root_id="root", recent_limit=2)
    failed = tree.add_node("移动到用户身边", node_id="move")
    tree.add_observation({"obstacle": True})
    tree.add_observation({"distance_cm": 15})
    tree.add_observation({"distance_cm": 12})

    alternative = tree.mark_branch_failed(
        failed.id,
        "前方有障碍，移动分支失败",
        alternative_goal="原地语音提醒",
    )

    assert [node.id for node in tree.active_path()] == ["root", alternative.id]
    assert tree.failed_summaries() == ("前方有障碍，移动分支失败",)
    assert tree.context_view()["recent_observations"] == []
    assert tree.nodes["move"].status is TaskNodeStatus.FAILED


def test_task_state_completed_summary_and_json_round_trip(tmp_path) -> None:
    tree = TaskStateTree("检查房间", root_id="root")
    done = tree.add_node("二次检查明火", node_id="inspect")
    tree.add_tool_result({"confidence": 0.91})
    tree.update_status(done.id, TaskNodeStatus.DONE, summary="确认存在明火")
    path = tmp_path / "task-state.json"

    tree.save(path)
    restored = TaskStateTree.load(path)

    assert restored.completed_summaries() == ("确认存在明火",)
    assert restored.nodes["inspect"].tool_results == [{"confidence": 0.91}]
    assert json.loads(path.read_text(encoding="utf-8"))["root_id"] == "root"
