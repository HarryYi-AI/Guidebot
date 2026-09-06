import json

from guidebot.eval import RuntimeEvalRunner, core_eval_cases


def test_core_eval_suite_passes() -> None:
    report = RuntimeEvalRunner().run(core_eval_cases())

    assert report.score == 1.0
    assert report.passed == len(core_eval_cases())
    assert report.failed == 0


def test_event_log_replay_aggregates_intents(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    events = (
        {"event_type": "user.text", "source": "test", "payload": {"text": "讲个故事"}},
        {
            "event_type": "scene.detected",
            "source": "test",
            "payload": {"label": "fire"},
        },
    )
    path.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")

    report = RuntimeEvalRunner().replay(path)

    assert report.events == 2
    assert report.scheduled_tasks == 2
    assert report.intent_counts == {"chat": 1, "safety_fire_alert": 1}
