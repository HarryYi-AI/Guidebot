"""DashScope Qwen Omni Realtime adapter with no embedded credentials."""

from __future__ import annotations

import asyncio
import base64
import os
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

from ..models import AudioFrame

BEIJING_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"


@dataclass(frozen=True, slots=True)
class DashScopeRealtimeConfig:
    model: str = "qwen3.5-omni-plus-realtime"
    url: str = BEIJING_URL
    voice: str = "Tina"
    instructions: str = (
        "你是 Guidebot，一只温暖、机灵的桌面机器人。回答自然简洁，不确定时坦诚说明。"
        "涉及移动或设备控制时只提出结构化请求，不声称动作已经完成。"
    )
    enable_search: bool = True
    search_sources: bool = True
    turn_detection: str = "semantic_vad"
    turn_detection_threshold: float | None = None
    turn_detection_silence_duration_ms: int | None = None
    connect_retries: int = 3
    connect_retry_delay_seconds: float = 2.0


@dataclass(frozen=True, slots=True)
class RealtimeEvent:
    type: str
    text: str = ""
    payload: dict[str, Any] | None = None


class _CallbackBridge:
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        audio_queue: asyncio.Queue[bytes | None],
        event_queue: asyncio.Queue[RealtimeEvent | None],
    ) -> None:
        self._loop = loop
        self._audio_queue = audio_queue
        self._event_queue = event_queue

    def on_open(self) -> None:
        self._emit(RealtimeEvent("session.open"))

    def on_close(self, close_status_code: int, close_msg: str) -> None:
        self._emit(
            RealtimeEvent(
                "session.close",
                payload={"code": close_status_code, "message": close_msg},
            )
        )

    def on_event(self, response: dict[str, Any]) -> None:
        event_type = response.get("type", "unknown")
        if event_type == "response.audio.delta":
            try:
                audio = base64.b64decode(response.get("delta", ""), validate=True)
            except (ValueError, TypeError):
                self._emit(RealtimeEvent("error", "invalid audio payload", response))
            else:
                if audio:
                    self._loop.call_soon_threadsafe(self._audio_queue.put_nowait, audio)
            return

        text = str(
            response.get("transcript")
            or response.get("text", "") + response.get("stash", "")
            or response.get("delta", "")
        )
        self._emit(RealtimeEvent(event_type, text, response))

    def _emit(self, event: RealtimeEvent) -> None:
        self._loop.call_soon_threadsafe(self._event_queue.put_nowait, event)


ConversationFactory = Callable[[str, object, str], Any]


class DashScopeRealtimeSession:
    """Async facade over the callback/thread based DashScope Python SDK."""

    def __init__(
        self,
        config: DashScopeRealtimeConfig | None = None,
        *,
        api_key: str | None = None,
        conversation_factory: ConversationFactory | None = None,
    ) -> None:
        self.config = config or DashScopeRealtimeConfig()
        self._api_key = api_key
        self._factory = conversation_factory
        self._audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._event_queue: asyncio.Queue[RealtimeEvent | None] = asyncio.Queue()
        self._conversation: Any = None
        self._closed = False

    async def connect(self) -> None:
        if self._conversation is not None:
            return
        api_key = self._api_key or os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is not set")

        loop = asyncio.get_running_loop()
        callback = _CallbackBridge(loop, self._audio_queue, self._event_queue)
        if self._factory is None:
            conversation, audio_modality, text_modality = self._sdk_conversation(
                api_key, callback
            )
        else:
            conversation = self._factory(self.config.model, callback, self.config.url)
            audio_modality, text_modality = "audio", "text"
        await self._connect_with_retry(conversation)
        self._conversation = conversation
        settings: dict[str, Any] = {
            "output_modalities": [audio_modality, text_modality],
            "voice": self.config.voice,
            "instructions": self.config.instructions,
            "enable_turn_detection": True,
            "turn_detection_type": self.config.turn_detection,
        }
        if self.config.turn_detection_threshold is not None:
            settings["turn_detection_threshold"] = self.config.turn_detection_threshold
        if self.config.turn_detection_silence_duration_ms is not None:
            settings["turn_detection_silence_duration_ms"] = (
                self.config.turn_detection_silence_duration_ms
            )
        if self.config.enable_search:
            settings.update(
                enable_search=True,
                search_options={"enable_source": self.config.search_sources},
            )
        await asyncio.to_thread(conversation.update_session, **settings)

    async def _connect_with_retry(self, conversation: Any) -> None:
        attempts = max(1, self.config.connect_retries)
        last_error: BaseException | None = None
        for attempt in range(1, attempts + 1):
            try:
                await asyncio.to_thread(conversation.connect)
                return
            except Exception as exc:  # noqa: BLE001 - SDK raises provider-specific errors.
                last_error = exc
                if attempt >= attempts:
                    break
                await asyncio.sleep(self.config.connect_retry_delay_seconds)
        raise RuntimeError(
            "DashScope realtime websocket connection failed after "
            f"{attempts} attempts. Please check Raspberry Pi network, DNS, firewall, "
            "system time, and whether wss://dashscope.aliyuncs.com is reachable."
        ) from last_error

    def _sdk_conversation(self, api_key: str, callback: object) -> tuple[Any, Any, Any]:
        try:
            import dashscope
            from dashscope.audio.qwen_omni import MultiModality, OmniRealtimeConversation
        except ImportError as exc:
            raise RuntimeError(
                "DashScope support is not installed; run pip install 'guidebot[voice-qwen]'"
            ) from exc
        dashscope.api_key = api_key
        conversation = OmniRealtimeConversation(
            model=self.config.model,
            callback=callback,
            url=self.config.url,
        )
        return conversation, MultiModality.AUDIO, MultiModality.TEXT

    async def send_audio(self, frame: AudioFrame) -> None:
        if (frame.sample_rate, frame.channels, frame.sample_width_bytes) != (16_000, 1, 2):
            raise ValueError("Qwen Realtime input must be 16 kHz mono PCM16")
        if self._conversation is None:
            await self.connect()
        encoded = base64.b64encode(frame.pcm).decode("ascii")
        await asyncio.to_thread(self._conversation.append_audio, encoded)

    async def receive_audio(self) -> AsyncIterator[bytes]:
        while True:
            chunk = await self._audio_queue.get()
            if chunk is None:
                return
            yield chunk

    async def receive_events(self) -> AsyncIterator[RealtimeEvent]:
        while True:
            event = await self._event_queue.get()
            if event is None:
                return
            yield event

    async def interrupt(self) -> None:
        if self._conversation is None:
            return
        cancel = getattr(self._conversation, "cancel_response", None)
        if cancel is not None:
            await asyncio.to_thread(cancel)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._conversation is not None:
            await asyncio.to_thread(self._conversation.close)
        await self._audio_queue.put(None)
        await self._event_queue.put(None)
