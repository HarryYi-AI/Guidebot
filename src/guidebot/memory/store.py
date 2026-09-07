"""SQLite structured store for temporal Guidebot memories."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import TypeVar, cast

from ..logbook import to_jsonable
from ..models import utc_now
from .schemas import (
    BoundaryMemory,
    EpisodeMemory,
    FactMemory,
    MemoryObject,
    MemoryStatus,
    PreferenceMemory,
    RelationshipMemory,
    SkillCandidate,
    SkillEvidenceMemory,
    TemporaryStateMemory,
)


MemoryT = TypeVar("MemoryT", bound=MemoryObject)

_TABLES = {
    EpisodeMemory: "episodes",
    FactMemory: "facts",
    TemporaryStateMemory: "states",
    PreferenceMemory: "preferences",
    BoundaryMemory: "boundaries",
    RelationshipMemory: "relationship_states",
    SkillEvidenceMemory: "skill_evidence",
    SkillCandidate: "skill_candidates",
}
_DATETIME_FIELDS = {
    "timestamp",
    "valid_from",
    "valid_to",
    "expires_at",
    "created_at",
}
_TUPLE_FIELDS = {"evidence_episode_ids", "procedure", "proposed_procedure"}


class StructuredMemoryStore:
    """Owns persistence; callers use MemoryService instead of raw SQL."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self._create_schema()

    def close(self) -> None:
        self.connection.close()

    def add(self, memory: MemoryT) -> MemoryT:
        table = _TABLES[type(memory)]
        payload = json.dumps(to_jsonable(memory), ensure_ascii=False, sort_keys=True)
        key = _memory_key(memory)
        timestamp = _memory_time(memory)
        valid_from = getattr(memory, "valid_from", None)
        valid_to = getattr(memory, "valid_to", None)
        self.connection.execute(
            f"""INSERT OR REPLACE INTO {table}
            (memory_id, user_id, status, memory_key, timestamp, valid_from, valid_to, payload)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",  # noqa: S608 - table comes from fixed map.
            (
                memory.memory_id,
                memory.user_id,
                memory.status.value,
                key,
                timestamp.isoformat(),
                valid_from.isoformat() if valid_from else None,
                valid_to.isoformat() if valid_to else None,
                payload,
            ),
        )
        self.connection.commit()
        return memory

    def get(self, memory_type: type[MemoryT], memory_id: str) -> MemoryT | None:
        table = _TABLES[memory_type]
        row = self.connection.execute(
            f"SELECT payload FROM {table} WHERE memory_id = ?",  # noqa: S608
            (memory_id,),
        ).fetchone()
        return _decode(memory_type, row["payload"]) if row else None

    def list(
        self,
        memory_type: type[MemoryT],
        user_id: str,
        *,
        status: MemoryStatus | None = None,
        key: str | None = None,
        limit: int | None = None,
    ) -> tuple[MemoryT, ...]:
        table = _TABLES[memory_type]
        clauses = ["user_id = ?"]
        params: list[object] = [user_id]
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value)
        if key is not None:
            clauses.append("memory_key = ?")
            params.append(key)
        sql = f"SELECT payload FROM {table} WHERE {' AND '.join(clauses)} ORDER BY timestamp DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(max(0, limit))
        rows = self.connection.execute(sql, params).fetchall()
        return tuple(_decode(memory_type, row["payload"]) for row in rows)

    def set_status(
        self,
        memory_type: type[MemoryT],
        memory_id: str,
        status: MemoryStatus,
        *,
        valid_to: datetime | None = None,
    ) -> MemoryT:
        current = self.get(memory_type, memory_id)
        if current is None:
            raise KeyError(memory_id)
        changes: dict[str, object] = {"status": status}
        if hasattr(current, "valid_to"):
            changes["valid_to"] = valid_to or utc_now()
        updated = replace(current, **changes)
        return self.add(cast(MemoryT, updated))

    def expire_states(self, user_id: str, *, now: datetime | None = None) -> None:
        current = now or utc_now()
        for state in self.list(
            TemporaryStateMemory,
            user_id,
            status=MemoryStatus.ACTIVE,
        ):
            if state.expires_at <= current:
                self.set_status(TemporaryStateMemory, state.memory_id, MemoryStatus.EXPIRED)

    def _create_schema(self) -> None:
        for table in _TABLES.values():
            self.connection.execute(
                f"""CREATE TABLE IF NOT EXISTS {table} (
                    memory_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    memory_key TEXT,
                    timestamp TEXT NOT NULL,
                    valid_from TEXT,
                    valid_to TEXT,
                    payload TEXT NOT NULL
                )"""  # noqa: S608
            )
            self.connection.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{table}_user_status "  # noqa: S608
                f"ON {table}(user_id, status)"
            )
            self.connection.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{table}_key ON {table}(memory_key)"  # noqa: S608
            )
            self.connection.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{table}_time ON {table}(timestamp)"  # noqa: S608
            )
            self.connection.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{table}_valid "  # noqa: S608
                f"ON {table}(valid_from, valid_to)"
            )
        self.connection.commit()


def _memory_key(memory: MemoryObject) -> str | None:
    return cast(
        str | None,
        getattr(memory, "key", None)
        or getattr(memory, "state", None)
        or getattr(memory, "skill_name", None)
        or getattr(memory, "name", None),
    )


def _memory_time(memory: MemoryObject) -> datetime:
    return cast(
        datetime,
        getattr(memory, "timestamp", None)
        or getattr(memory, "valid_from", None)
        or getattr(memory, "created_at", None)
        or utc_now(),
    )


def _decode(memory_type: type[MemoryT], payload: str) -> MemoryT:
    values = json.loads(payload)
    for key in _DATETIME_FIELDS:
        if values.get(key):
            values[key] = datetime.fromisoformat(values[key])
    for key in _TUPLE_FIELDS:
        if key in values:
            values[key] = tuple(values[key])
    values["status"] = MemoryStatus(values["status"])
    return memory_type(**values)
