"""Append-only JSONL storage for complete task trajectories."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..logbook import to_jsonable
from ..models import utc_now


@dataclass(frozen=True, slots=True)
class Episode:
    task: str
    trajectory: tuple[Any, ...]
    outcome: str
    success: bool
    episode_id: str = field(default_factory=lambda: uuid4().hex)
    timestamp: datetime = field(default_factory=utc_now)


class EpisodicMemory:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, episode: Episode) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(to_jsonable(episode), ensure_ascii=False) + "\n")

    def read_all(self) -> tuple[dict[str, Any], ...]:
        if not self.path.exists():
            return ()
        return tuple(
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
