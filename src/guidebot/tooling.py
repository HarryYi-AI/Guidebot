"""Auditable tool contracts and bounded execution for runtime skills."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from enum import Enum
from time import perf_counter
from typing import Any


class ToolPermission(str, Enum):
    READ_ONLY = "read_only"
    COMMUNICATE = "communicate"
    WRITE_STATE = "write_state"
    EXTERNAL_SIDE_EFFECT = "external_side_effect"
    PHYSICAL = "physical"
    SAFETY_CRITICAL = "safety_critical"


class ErrorSemantics(str, Enum):
    FAIL_CLOSED = "fail_closed"
    RETRYABLE = "retryable"
    BEST_EFFORT = "best_effort"


class ToolStatus(str, Enum):
    SUCCEEDED = "succeeded"
    INVALID_ARGUMENTS = "invalid_arguments"
    TIMED_OUT = "timed_out"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ToolContract:
    """Schema, authority, deadline, and failure behavior outside model control."""

    name: str
    permission: ToolPermission
    input_schema: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    required_fields: tuple[str, ...] = ()
    timeout_s: float = 5.0
    error_semantics: ErrorSemantics = ErrorSemantics.FAIL_CLOSED
    max_attempts: int = 1
    idempotent: bool = False
    audit: bool = True

    def __post_init__(self) -> None:
        if self.timeout_s <= 0:
            raise ValueError("tool timeout must be positive")
        if self.max_attempts < 1:
            raise ValueError("tool max_attempts must be at least one")
        if self.max_attempts > 1 and not self.idempotent:
            raise ValueError("only idempotent tools may be retried")

    def validate(self, payload: Mapping[str, Any]) -> tuple[str, ...]:
        errors = [f"missing required field: {name}" for name in self.required_fields if name not in payload]
        for name, expected in self.input_schema.items():
            if name not in payload:
                continue
            if not _matches_type(payload[name], expected):
                errors.append(
                    f"field {name} expected {'|'.join(expected)}, got {type(payload[name]).__name__}"
                )
        return tuple(errors)


@dataclass(frozen=True, slots=True)
class ToolExecution:
    tool_name: str
    status: ToolStatus
    permission: ToolPermission
    attempts: int
    latency_ms: float
    output: dict[str, Any] | None = None
    error_code: str | None = None
    error_message: str | None = None


class ToolDispatcher:
    """Validate and execute at most the bounded attempts declared by the contract."""

    def execute(
        self,
        contract: ToolContract,
        payload: Mapping[str, Any],
        invoke: Callable[[], dict[str, Any]],
    ) -> ToolExecution:
        errors = contract.validate(payload)
        if errors:
            return ToolExecution(
                contract.name,
                ToolStatus.INVALID_ARGUMENTS,
                contract.permission,
                0,
                0.0,
                error_code="schema_validation",
                error_message="; ".join(errors),
            )

        attempts_allowed = contract.max_attempts if contract.idempotent else 1
        total_started = perf_counter()
        last_error: Exception | None = None
        attempts_made = 0
        for attempt in range(1, attempts_allowed + 1):
            attempts_made = attempt
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="guidebot-tool")
            future = executor.submit(invoke)
            try:
                output = future.result(timeout=contract.timeout_s)
            except FutureTimeoutError:
                future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                return ToolExecution(
                    contract.name,
                    ToolStatus.TIMED_OUT,
                    contract.permission,
                    attempt,
                    (perf_counter() - total_started) * 1_000,
                    error_code="timeout",
                    error_message=f"tool exceeded {contract.timeout_s:.3f}s deadline",
                )
            except Exception as exc:  # noqa: BLE001 - contract defines provider error semantics.
                executor.shutdown(wait=True, cancel_futures=True)
                last_error = exc
                if contract.error_semantics is ErrorSemantics.RETRYABLE and attempt < attempts_allowed:
                    continue
                break
            else:
                executor.shutdown(wait=True, cancel_futures=True)
                if not isinstance(output, dict):
                    return ToolExecution(
                        contract.name,
                        ToolStatus.FAILED,
                        contract.permission,
                        attempt,
                        (perf_counter() - total_started) * 1_000,
                        error_code="invalid_tool_result",
                        error_message="tool result must be a dictionary",
                    )
                return ToolExecution(
                    contract.name,
                    ToolStatus.SUCCEEDED,
                    contract.permission,
                    attempt,
                    (perf_counter() - total_started) * 1_000,
                    output=output,
                )

        return ToolExecution(
            contract.name,
            ToolStatus.FAILED,
            contract.permission,
            attempts_made,
            (perf_counter() - total_started) * 1_000,
            error_code="execution_error",
            error_message=str(last_error) if last_error else "tool execution failed",
        )


_JSON_TYPES: dict[str, type[Any] | tuple[type[Any], ...]] = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "object": dict,
    "array": (list, tuple),
    "null": type(None),
}


def _matches_type(value: Any, expected: tuple[str, ...]) -> bool:
    for type_name in expected:
        python_type = _JSON_TYPES.get(type_name)
        if python_type is None:
            raise ValueError(f"unsupported schema type: {type_name}")
        if type_name in {"number", "integer"} and isinstance(value, bool):
            continue
        if isinstance(value, python_type):
            return True
    return False
