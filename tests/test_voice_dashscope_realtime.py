import asyncio
import base64

import pytest

from guidebot.voice.models import AudioFrame
from guidebot.voice.providers import DashScopeRealtimeConfig, DashScopeRealtimeSession


class FakeConversation:
    def __init__(self, callback):
        self.callback = callback
        self.connected = False
        self.closed = False
        self.settings = {}
        self.audio = []
        self.cancelled = False

    def connect(self):
        self.connected = True
        self.callback.on_open()

    def update_session(self, **settings):
        self.settings = settings

    def append_audio(self, audio):
        self.audio.append(audio)

    def cancel_response(self):
        self.cancelled = True

    def close(self):
        self.closed = True


class FlakyConversation(FakeConversation):
    def __init__(self, callback, failures: int):
        super().__init__(callback)
        self.failures = failures
        self.connect_attempts = 0

    def connect(self):
        self.connect_attempts += 1
        if self.connect_attempts <= self.failures:
            raise TimeoutError("websocket connection could not established within 5s")
        super().connect()


@pytest.mark.asyncio
async def test_session_streams_audio_and_configures_semantic_vad(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-only")
    holder = {}

    def factory(model, callback, url):
        holder["model"] = model
        holder["url"] = url
        holder["conversation"] = FakeConversation(callback)
        return holder["conversation"]

    session = DashScopeRealtimeSession(conversation_factory=factory)
    await session.connect()
    conversation = holder["conversation"]
    assert conversation.settings["turn_detection_type"] == "semantic_vad"
    assert conversation.settings["enable_search"] is True

    frame = AudioFrame(b"\x01\x00" * 160)
    await session.send_audio(frame)
    assert base64.b64decode(conversation.audio[0]) == frame.pcm

    conversation.callback.on_event(
        {"type": "response.audio.delta", "delta": base64.b64encode(b"voice").decode()}
    )
    assert await anext(session.receive_audio()) == b"voice"
    await session.interrupt()
    assert conversation.cancelled
    await session.close()
    assert conversation.closed


@pytest.mark.asyncio
async def test_session_requires_environment_key(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    session = DashScopeRealtimeSession(conversation_factory=lambda *_: None)
    with pytest.raises(RuntimeError, match="DASHSCOPE_API_KEY"):
        await session.connect()


@pytest.mark.asyncio
async def test_session_rejects_wrong_audio_format(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-only")
    session = DashScopeRealtimeSession(
        DashScopeRealtimeConfig(enable_search=False),
        conversation_factory=lambda _model, callback, _url: FakeConversation(callback),
    )
    with pytest.raises(ValueError, match="16 kHz"):
        await session.send_audio(AudioFrame(b"\0\0", sample_rate=24_000))
    await session.close()


@pytest.mark.asyncio
async def test_callback_events_cross_thread_safely(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-only")
    holder = {}

    def factory(_model, callback, _url):
        holder["callback"] = callback
        return FakeConversation(callback)

    session = DashScopeRealtimeSession(conversation_factory=factory)
    await session.connect()
    holder["callback"].on_event(
        {"type": "response.audio_transcript.done", "transcript": "你好"}
    )
    event = await asyncio.wait_for(anext(session.receive_events()), 0.2)
    if event.type == "session.open":
        event = await asyncio.wait_for(anext(session.receive_events()), 0.2)
    assert event.text == "你好"
    await session.close()


@pytest.mark.asyncio
async def test_session_retries_transient_connection_timeout(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-only")
    holder = {}

    def factory(_model, callback, _url):
        holder["conversation"] = FlakyConversation(callback, failures=1)
        return holder["conversation"]

    session = DashScopeRealtimeSession(
        DashScopeRealtimeConfig(connect_retries=2, connect_retry_delay_seconds=0),
        conversation_factory=factory,
    )
    await session.connect()
    assert holder["conversation"].connect_attempts == 2
    assert holder["conversation"].connected is True
    await session.close()


@pytest.mark.asyncio
async def test_session_reports_connection_timeout_after_retries(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-only")

    session = DashScopeRealtimeSession(
        DashScopeRealtimeConfig(connect_retries=2, connect_retry_delay_seconds=0),
        conversation_factory=lambda _model, callback, _url: FlakyConversation(
            callback,
            failures=2,
        ),
    )
    with pytest.raises(RuntimeError, match="after 2 attempts"):
        await session.connect()
