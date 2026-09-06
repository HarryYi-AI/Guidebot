import time

from guidebot.tooling import (
    ErrorSemantics,
    ToolContract,
    ToolDispatcher,
    ToolPermission,
    ToolStatus,
)


def test_schema_validation_prevents_tool_execution() -> None:
    called = False

    def invoke() -> dict:
        nonlocal called
        called = True
        return {}

    contract = ToolContract(
        "alarm.set",
        ToolPermission.WRITE_STATE,
        input_schema={"time": ("string",)},
        required_fields=("time",),
    )

    result = ToolDispatcher().execute(contract, {"time": 700}, invoke)

    assert result.status is ToolStatus.INVALID_ARGUMENTS
    assert result.error_code == "schema_validation"
    assert called is False


def test_retry_is_bounded_and_only_for_idempotent_tool() -> None:
    attempts = 0

    def invoke() -> dict:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError("temporary")
        return {"module": "test", "action": "read"}

    contract = ToolContract(
        "sensor.read",
        ToolPermission.READ_ONLY,
        error_semantics=ErrorSemantics.RETRYABLE,
        max_attempts=2,
        idempotent=True,
    )

    result = ToolDispatcher().execute(contract, {}, invoke)

    assert result.status is ToolStatus.SUCCEEDED
    assert result.attempts == 2
    assert attempts == 2


def test_timeout_has_explicit_error_semantics() -> None:
    contract = ToolContract("slow.read", ToolPermission.READ_ONLY, timeout_s=0.001)

    result = ToolDispatcher().execute(
        contract,
        {},
        lambda: (time.sleep(0.02) or {"module": "test", "action": "read"}),
    )

    assert result.status is ToolStatus.TIMED_OUT
    assert result.error_code == "timeout"


def test_non_mapping_tool_result_is_rejected() -> None:
    contract = ToolContract("bad.read", ToolPermission.READ_ONLY)

    result = ToolDispatcher().execute(contract, {}, lambda: "not structured")  # type: ignore[arg-type]

    assert result.status is ToolStatus.FAILED
    assert result.error_code == "invalid_tool_result"
