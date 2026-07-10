"""JSONL logging for Guidebot runtime traces."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class RuntimeLogger:
    def __init__(self, log_dir: str | Path = "logs") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def event(self, payload: Any) -> None:
        self._write("events.jsonl", payload)

    def intent(self, payload: Any) -> None:
        self._write("intents.jsonl", payload)

    def task(self, payload: Any) -> None:
        self._write("tasks.jsonl", payload)

    def _write(self, filename: str, payload: Any) -> None:
        with (self.log_dir / filename).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(to_jsonable(payload), ensure_ascii=False) + "\n")


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value
