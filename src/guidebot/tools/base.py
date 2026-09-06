"""Minimal asynchronous Tool contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolResult:
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class Tool(ABC):
    """A registered capability; planner output can only reference its name/schema."""

    name: str
    description: str
    schema: dict[str, Any]
    physical: bool = False

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the wrapped capability and return a structured observation."""


def validate_json_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> tuple[str, ...]:
    """Validate the small JSON-Schema subset used by Guidebot tools."""
    if schema.get("type", "object") != "object":
        raise ValueError("Guidebot tool schemas must describe an object")
    properties = schema.get("properties", {})
    required = schema.get("required", ())
    additional = schema.get("additionalProperties", False)
    errors = [f"missing required argument: {name}" for name in required if name not in arguments]
    if not additional:
        errors.extend(f"unknown argument: {name}" for name in arguments if name not in properties)
    for name, value in arguments.items():
        spec = properties.get(name)
        if not isinstance(spec, dict):
            continue
        expected = spec.get("type")
        if expected and not _matches_json_type(value, expected):
            errors.append(f"argument {name} must be {expected}")
        if "minimum" in spec and isinstance(value, (int, float)) and value < spec["minimum"]:
            errors.append(f"argument {name} is below minimum")
        if "maximum" in spec and isinstance(value, (int, float)) and value > spec["maximum"]:
            errors.append(f"argument {name} is above maximum")
    return tuple(errors)


def _matches_json_type(value: Any, expected: str | list[str]) -> bool:
    names = [expected] if isinstance(expected, str) else expected
    types = {
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "object": dict,
        "array": list,
        "null": type(None),
    }
    return any(
        name in types
        and not (name in {"number", "integer"} and isinstance(value, bool))
        and isinstance(value, types[name])
        for name in names
    )
