import pytest

from guidebot.voice.commands import VoiceIntentRouter


class FakeVolume:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int | None]] = []

    async def set_percent(self, percent: int) -> None:
        self.calls.append(("set", percent))

    async def change_percent(self, delta: int) -> None:
        self.calls.append(("change", delta))

    async def mute(self) -> None:
        self.calls.append(("mute", None))

    async def unmute(self) -> None:
        self.calls.append(("unmute", None))


@pytest.mark.asyncio
async def test_voice_router_sets_explicit_volume() -> None:
    volume = FakeVolume()
    router = VoiceIntentRouter(volume)

    result = await router.route("请把音量调到 50%")

    assert result is not None
    assert result.intent == "volume.set"
    assert volume.calls == [("set", 50)]


@pytest.mark.asyncio
async def test_voice_router_changes_volume_relatively() -> None:
    volume = FakeVolume()
    router = VoiceIntentRouter(volume, volume_step=15)

    result = await router.route("声音小一点")

    assert result is not None
    assert result.intent == "volume.down"
    assert volume.calls == [("change", -15)]


@pytest.mark.asyncio
async def test_voice_router_handles_mute_and_unmute() -> None:
    volume = FakeVolume()
    router = VoiceIntentRouter(volume)

    muted = await router.route("静音")
    unmuted = await router.route("取消静音")

    assert muted is not None
    assert unmuted is not None
    assert volume.calls == [("mute", None), ("unmute", None)]


@pytest.mark.asyncio
async def test_voice_router_ignores_general_chat() -> None:
    volume = FakeVolume()
    router = VoiceIntentRouter(volume)

    result = await router.route("今天天气怎么样")

    assert result is None
    assert volume.calls == []
