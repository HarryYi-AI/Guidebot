"""Lightweight execution-memory tree for tracking active and failed task branches."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4


class TaskNodeStatus(str, Enum):
    ACTIVE = "active"
    DONE = "done"
    FAILED = "failed"
    ABANDONED = "abandoned"


@dataclass(slots=True)
class TaskNode:
    id: str
    parent_id: str | None
    goal: str
    status: TaskNodeStatus = TaskNodeStatus.ACTIVE
    summary: str = ""
    recent_observations: list[Any] = field(default_factory=list)
    tool_results: list[Any] = field(default_factory=list)


class TaskStateTree:
    """Tracks one active execution branch while retaining compact branch outcomes."""

    def __init__(
        self,
        root_goal: str,
        *,
        root_id: str | None = None,
        recent_limit: int = 6,
    ) -> None:
        if not root_goal.strip():
            raise ValueError("root goal must not be empty")
        if recent_limit < 1:
            raise ValueError("recent_limit must be positive")
        root = TaskNode(root_id or uuid4().hex, None, root_goal.strip())
        self.nodes: dict[str, TaskNode] = {root.id: root}
        self.root_id = root.id
        self.current_active_id = root.id
        self.recent_limit = recent_limit

    @property
    def current(self) -> TaskNode:
        return self.nodes[self.current_active_id]

    def add_node(
        self,
        goal: str,
        *,
        parent_id: str | None = None,
        status: TaskNodeStatus = TaskNodeStatus.ACTIVE,
        summary: str = "",
        node_id: str | None = None,
    ) -> TaskNode:
        parent = parent_id or self.current_active_id
        if parent not in self.nodes:
            raise KeyError(f"unknown parent node: {parent}")
        if not goal.strip():
            raise ValueError("node goal must not be empty")
        identifier = node_id or uuid4().hex
        if identifier in self.nodes:
            raise ValueError(f"duplicate task node: {identifier}")
        node = TaskNode(identifier, parent, goal.strip(), status, summary)
        self.nodes[node.id] = node
        if status is TaskNodeStatus.ACTIVE:
            self.current_active_id = node.id
        return node

    def update_status(
        self,
        node_id: str,
        status: TaskNodeStatus,
        *,
        summary: str | None = None,
    ) -> TaskNode:
        node = self.nodes[node_id]
        node.status = status
        if summary is not None:
            node.summary = summary
        if status is TaskNodeStatus.ACTIVE:
            self.current_active_id = node_id
        elif node_id == self.current_active_id:
            parent = self.nodes.get(node.parent_id or "")
            if parent is not None:
                parent.status = TaskNodeStatus.ACTIVE
                self.current_active_id = parent.id
        return node

    def add_observation(self, observation: Any, *, node_id: str | None = None) -> None:
        node = self.nodes[node_id or self.current_active_id]
        node.recent_observations.append(observation)
        del node.recent_observations[: -self.recent_limit]

    def add_tool_result(self, result: Any, *, node_id: str | None = None) -> None:
        node = self.nodes[node_id or self.current_active_id]
        node.tool_results.append(result)
        del node.tool_results[: -self.recent_limit]

    def active_path(self) -> tuple[TaskNode, ...]:
        path: list[TaskNode] = []
        seen: set[str] = set()
        node: TaskNode | None = self.current
        while node is not None:
            if node.id in seen:
                raise ValueError("cycle detected in task state tree")
            seen.add(node.id)
            path.append(node)
            node = self.nodes.get(node.parent_id or "")
        path.reverse()
        if not path or path[0].id != self.root_id:
            raise ValueError("active branch is disconnected from root")
        return tuple(path)

    def completed_summaries(self) -> tuple[str, ...]:
        return tuple(
            node.summary or node.goal
            for node in self.nodes.values()
            if node.status is TaskNodeStatus.DONE
        )

    def failed_summaries(self) -> tuple[str, ...]:
        return tuple(
            node.summary or node.goal
            for node in self.nodes.values()
            if node.status in {TaskNodeStatus.FAILED, TaskNodeStatus.ABANDONED}
        )

    def mark_branch_failed(
        self,
        node_id: str,
        summary: str,
        *,
        alternative_id: str | None = None,
        alternative_goal: str | None = None,
    ) -> TaskNode:
        failed = self.nodes[node_id]
        failed.status = TaskNodeStatus.FAILED
        failed.summary = summary
        for descendant in self._descendants(node_id):
            if descendant.status is TaskNodeStatus.ACTIVE:
                descendant.status = TaskNodeStatus.ABANDONED
        parent_id = failed.parent_id or self.root_id
        if alternative_id is not None:
            if alternative_id == node_id:
                raise ValueError("failed node cannot be its own alternative")
            alternative = self.nodes[alternative_id]
            if alternative.parent_id != failed.parent_id:
                raise ValueError("alternative branch must share the failed node's parent")
            alternative.status = TaskNodeStatus.ACTIVE
        elif alternative_goal is not None:
            alternative = self.add_node(alternative_goal, parent_id=parent_id)
        else:
            alternative = self.nodes[parent_id]
            alternative.status = TaskNodeStatus.ACTIVE
        self.current_active_id = alternative.id
        return alternative

    def context_view(self) -> dict[str, Any]:
        """Return only active path, current observations and compact failed branches."""
        return {
            "active_task_path": [
                {
                    "id": node.id,
                    "goal": node.goal,
                    "status": node.status.value,
                    "summary": node.summary,
                }
                for node in self.active_path()
            ],
            "recent_observations": list(self.current.recent_observations),
            "failed_branches": list(self.failed_summaries()),
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "root_id": self.root_id,
            "current_active_id": self.current_active_id,
            "recent_limit": self.recent_limit,
            "nodes": [
                {**asdict(node), "status": node.status.value} for node in self.nodes.values()
            ],
        }
        destination.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> TaskStateTree:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        required = {"root_id", "current_active_id", "recent_limit", "nodes"}
        if not isinstance(payload, dict) or set(payload) != required:
            raise ValueError("invalid task state payload")
        node_items = payload["nodes"]
        if not isinstance(node_items, list) or not node_items:
            raise ValueError("task state requires at least one node")
        decoded = {
            str(item["id"]): TaskNode(
                id=str(item["id"]),
                parent_id=item.get("parent_id"),
                goal=str(item["goal"]),
                status=TaskNodeStatus(item["status"]),
                summary=str(item.get("summary", "")),
                recent_observations=list(item.get("recent_observations", ())),
                tool_results=list(item.get("tool_results", ())),
            )
            for item in node_items
        }
        root_id = str(payload["root_id"])
        current_id = str(payload["current_active_id"])
        if root_id not in decoded or current_id not in decoded:
            raise ValueError("task state references an unknown root or active node")
        if decoded[root_id].parent_id is not None or any(
            node.parent_id is not None and node.parent_id not in decoded
            for node in decoded.values()
        ):
            raise ValueError("task state contains an invalid parent reference")
        tree = cls(decoded[root_id].goal, root_id=root_id, recent_limit=int(payload["recent_limit"]))
        tree.nodes = decoded
        tree.current_active_id = current_id
        tree.active_path()
        return tree

    def _descendants(self, node_id: str) -> tuple[TaskNode, ...]:
        result: list[TaskNode] = []
        frontier = [node_id]
        while frontier:
            parent = frontier.pop()
            children = [node for node in self.nodes.values() if node.parent_id == parent]
            result.extend(children)
            frontier.extend(node.id for node in children)
        return tuple(result)
