import pytest

from guidebot.tools import SpeakTool, ToolRegistry, UnknownToolError


@pytest.mark.asyncio
async def test_tool_registry_register_and_execute() -> None:
    registry = ToolRegistry()
    registry.register(SpeakTool())

    result = await registry.execute("speak", text="起来活动一下吧")

    assert result.success is True
    assert result.data["spoken"] == "起来活动一下吧"
    assert registry.schemas()[0]["name"] == "speak"


@pytest.mark.asyncio
async def test_tool_registry_rejects_invalid_arguments() -> None:
    registry = ToolRegistry()
    registry.register(SpeakTool())

    result = await registry.execute("speak", wrong="value")

    assert result.success is False
    assert "missing required" in (result.error or "")


@pytest.mark.asyncio
async def test_unknown_tool_is_rejected() -> None:
    registry = ToolRegistry()

    with pytest.raises(UnknownToolError):
        await registry.execute("delete_everything")
