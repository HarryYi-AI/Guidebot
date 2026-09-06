"""Versioned stable memories with explicit supersession."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from ..models import utc_now


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    memory_id: str
    content: str
    importance: float
    created_at: datetime
    updated_at: datetime
    active: bool = True
    superseded_by: str | None = None


class LongTermMemory:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._records: dict[str, MemoryRecord] = {}
        self._load()

    def add(self, content: str, *, importance: float = 0.5) -> MemoryRecord:
        if not content.strip() or not 0 <= importance <= 1:
            raise ValueError("memory requires content and importance within 0..1")
        now = utc_now()
        record = MemoryRecord(uuid4().hex, content.strip(), importance, now, now)
        self._records[record.memory_id] = record
        self._persist()
        return record

    def update(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        importance: float | None = None,
    ) -> MemoryRecord:
        record = self._records[memory_id]
        if importance is not None and not 0 <= importance <= 1:
            raise ValueError("importance must be within 0..1")
        updated = replace(
            record,
            content=content.strip() if content is not None else record.content,
            importance=importance if importance is not None else record.importance,
            updated_at=utc_now(),
        )
        self._records[memory_id] = updated
        self._persist()
        return updated

    def supersede(
        self,
        memory_id: str,
        content: str,
        *,
        importance: float | None = None,
    ) -> MemoryRecord:
        previous = self._records[memory_id]
        replacement = self.add(
            content,
            importance=previous.importance if importance is None else importance,
        )
        self._records[memory_id] = replace(
            previous,
            active=False,
            superseded_by=replacement.memory_id,
            updated_at=utc_now(),
        )
        self._persist()
        return replacement

    def list_active(self) -> tuple[MemoryRecord, ...]:
        return tuple(record for record in self._records.values() if record.active)

    def get(self, memory_id: str) -> MemoryRecord:
        return self._records[memory_id]

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        for item in json.loads(self.path.read_text(encoding="utf-8")):
            item["created_at"] = datetime.fromisoformat(item["created_at"])
            item["updated_at"] = datetime.fromisoformat(item["updated_at"])
            record = MemoryRecord(**item)
            self._records[record.memory_id] = record

    def _persist(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = []
        for record in self._records.values():
            item = asdict(record)
            item["created_at"] = record.created_at.isoformat()
            item["updated_at"] = record.updated_at.isoformat()
            payload.append(item)
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
