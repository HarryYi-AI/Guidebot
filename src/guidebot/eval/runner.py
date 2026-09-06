"""Automatic regression evaluation and JSONL event replay."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from guidebot.events import Event
from guidebot.runtime import GuidebotRuntime

from .cases import EvalCase


@dataclass(frozen=True, slots=True)
class CaseResult:
    name: str
    passed: bool
    actual_intent: str
    actual_skill_id: str | None
    safety_allowed: bool | None
    reason: str
    final_status: str = "unknown"


@dataclass(frozen=True, slots=True)
class EvalReport:
    suite: str
    passed: int
    failed: int
    score: float
    cases: tuple[CaseResult, ...]


@dataclass(frozen=True, slots=True)
class ReplayReport:
    events: int
    scheduled_tasks: int
    blocked_tasks: int
    intent_counts: dict[str, int]


class RuntimeEvalRunner:
    """Runs every held-out case in an isolated runtime to avoid cooldown leakage."""

    def __init__(self, runtime_factory: Callable[[], GuidebotRuntime] = GuidebotRuntime) -> None:
        self.runtime_factory = runtime_factory

    def run(self, cases: Iterable[EvalCase], *, suite: str = "core") -> EvalReport:
        results = tuple(self._run_case(case) for case in cases)
        passed = sum(result.passed for result in results)
        total = len(results)
        return EvalReport(suite, passed, total - passed, passed / total if total else 0.0, results)

    def replay(self, path: str | Path) -> ReplayReport:
        runtime = self.runtime_factory()
        counts: Counter[str] = Counter()
        events = scheduled = blocked = 0
        try:
            for event in _read_events(path):
                trace = runtime.ingest(event)
                events += 1
                counts[trace.intent.intent_type.value] += 1
                scheduled += trace.task is not None
                blocked += bool(trace.action and trace.action.get("blocked"))
        finally:
            runtime.stop()
        return ReplayReport(events, scheduled, blocked, dict(sorted(counts.items())))

    def _run_case(self, case: EvalCase) -> CaseResult:
        runtime = self.runtime_factory()
        try:
            trace = runtime.ingest(case.event)
        except Exception as exc:  # noqa: BLE001 - eval must report arbitrary module failures.
            return CaseResult(case.name, False, "error", None, None, str(exc))
        finally:
            runtime.stop()
        skill_id = trace.task.skill_id if trace.task else None
        safety_allowed = trace.safety.allowed if trace.safety else None
        checks = (
            trace.intent.intent_type is case.expected_intent,
            skill_id == case.expected_skill_id,
            case.expected_safety_allowed is None
            or safety_allowed is case.expected_safety_allowed,
            trace.trajectory is not None and trace.trajectory.success,
        )
        passed = all(checks)
        reason = "passed" if passed else (
            f"expected intent={case.expected_intent.value}, skill={case.expected_skill_id}, "
            f"safety={case.expected_safety_allowed}"
        )
        return CaseResult(
            case.name,
            passed,
            trace.intent.intent_type.value,
            skill_id,
            safety_allowed,
            reason,
            trace.final_status,
        )


def _read_events(path: str | Path) -> Iterable[Event]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                yield Event(
                    event_type=str(payload["event_type"]),
                    source=str(payload.get("source", "replay")),
                    payload=dict(payload.get("payload", {})),
                    timestamp=_timestamp(payload.get("timestamp")),
                    confidence=float(payload.get("confidence", 1.0)),
                    priority_hint=int(payload.get("priority_hint", 0)),
                    event_id=str(payload.get("event_id", f"replay-{line_number}")),
                    session_id=(
                        str(payload["session_id"])
                        if payload.get("session_id") is not None
                        else None
                    ),
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid event JSONL at line {line_number}: {exc}") from exc


def _timestamp(value: object) -> datetime:
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    return datetime.now().astimezone()
