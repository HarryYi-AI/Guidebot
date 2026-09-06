"""Single allow-listed gateway for all Agent Runtime tool execution."""

from __future__ import annotations

from typing import Any

from .base import Tool, ToolResult, validate_json_arguments


class UnknownToolError(KeyError):
    pass


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if not tool.name or tool.name in self._tools:
            raise ValueError(f"duplicate or empty tool name: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise UnknownToolError(name) from exc

    def schemas(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.schema,
            }
            for tool in self._tools.values()
        )

    def validate(self, name: str, arguments: dict[str, Any]) -> tuple[str, ...]:
        return validate_json_arguments(self.get(name).schema, arguments)

    async def execute(self, name: str, **kwargs: Any) -> ToolResult:
        tool = self.get(name)
        errors = validate_json_arguments(tool.schema, kwargs)
        if errors:
            return ToolResult(False, error="; ".join(errors))
        try:
            result = await tool.execute(**kwargs)
        except Exception as exc:  # noqa: BLE001 - adapters normalize provider failures.
            return ToolResult(False, error=f"{type(exc).__name__}: {exc}")
        if not isinstance(result, ToolResult):
            return ToolResult(False, error="tool returned an invalid result type")
        return result

    def __contains__(self, name: str) -> bool:
        return name in self._tools
