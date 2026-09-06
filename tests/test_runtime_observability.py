import json

from guidebot.events import Event
from guidebot.logbook import RuntimeLogger
from guidebot.outcomes import OutcomeType
from guidebot.runtime import GuidebotRuntime
from guidebot.task_verification import TaskVerificationStatus
from guidebot.tooling import ToolStatus


def test_runtime_records_execution_verification_and_final_status(tmp_path) -> None:
    runtime = GuidebotRuntime(logger=RuntimeLogger(tmp_path))

    trace = runtime.ingest(
        Event("user.text", "voice", {"text": "讲个故事"}, session_id="session-1")
    )

    assert trace.trace_id == trace.event.event_id
    assert trace.session_id == "session-1"
    assert trace.execution is not None
    assert trace.execution.status is ToolStatus.SUCCEEDED
    assert trace.verification is not None
    assert trace.verification.status is TaskVerificationStatus.PASSED
    assert trace.final_status == "succeeded"
    assert trace.outcome_type is OutcomeType.EXECUTED
    assert trace.human_summary().startswith("[executed] [voice.chat] [")

    records = (tmp_path / "traces.jsonl").read_text(encoding="utf-8").splitlines()
    persisted = json.loads(records[-1])
    assert persisted["trace_id"] == trace.trace_id
    assert persisted["execution"]["tool_name"] == "voice.chat"
    assert persisted["verification"]["status"] == "passed"
    assert persisted["final_status"] == "succeeded"


def test_safety_rejection_is_a_terminal_observable_status() -> None:
    runtime = GuidebotRuntime()
    runtime.safety_state.active_safety_alert = True

    trace = runtime.ingest(Event("user.text", "voice", {"text": "讲个故事"}))

    assert trace.execution is None
    assert trace.verification is not None
    assert trace.verification.status is TaskVerificationStatus.NOT_RUN
    assert trace.final_status == "safety_rejected"
    assert trace.outcome_type is OutcomeType.FAILED
    assert trace.trajectory is not None
    assert trace.trajectory.success is False


def test_unknown_event_needing_no_action_is_successful() -> None:
    trace = GuidebotRuntime().ingest(Event("unknown", "test", {}))

    assert trace.final_status == "no_action_required"
    assert trace.outcome_type is OutcomeType.NO_ACTION_REQUIRED
    assert trace.trajectory is not None
    assert trace.trajectory.success is True
    assert trace.trajectory.reward.total == 0.0
