"""Full-duplex runtime joining a realtime provider to microphone and speaker."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from .interfaces import AudioPlayer, AudioSource, NativeSpeechSession
from .providers.dashscope_realtime import RealtimeEvent
from .session_control import TRANSCRIPT_DONE, WakeSleepController
from .commands import VoiceIntentRouter


class NativeVoiceRuntime:
    def __init__(
        self,
        source: AudioSource,
        player: AudioPlayer,
        session: NativeSpeechSession,
        on_event: Callable[[object], None] | None = None,
        session_controller: WakeSleepController | None = None,
        command_router: VoiceIntentRouter | None = None,
    ) -> None:
        self.source = source
        self.player = player
        self.session = session
        self.on_event = on_event
        self.session_controller = session_controller
        self.command_router = command_router
        self._responding = False

    async def run(self) -> None:
        connect = getattr(self.session, "connect", None)
        if connect is not None:
            await connect()
        tasks = [
            asyncio.create_task(self._capture(), name="voice-capture"),
            asyncio.create_task(self._playback(), name="voice-playback"),
            asyncio.create_task(self._events(), name="voice-events"),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self.player.stop()
            close_source = getattr(self.source, "close", None)
            if close_source is not None:
                await close_source()
            await self.session.close()

    async def _capture(self) -> None:
        while True:
            frame = await self.source.read()
            if frame is None:
                return
            await self.session.send_audio(frame)

    async def _playback(self) -> None:
        async for audio in self.session.receive_audio():
            if (
                self.session_controller is not None
                and not self.session_controller.allow_playback
            ):
                continue
            await self.player.play(audio)

    async def _events(self) -> None:
        async for event in self.session.receive_events():
            if isinstance(event, RealtimeEvent):
                if event.type == "response.created":
                    self._responding = True
                elif event.type == "response.done":
                    self._responding = False
                if event.type == "input_audio_buffer.speech_started":
                    await self.player.stop()
                    if self._responding:
                        await self.session.interrupt()
                if event.type == "error":
                    raise RuntimeError(event.text or "realtime provider error")
                if self.session_controller is not None:
                    decision = self.session_controller.update(event)
                    if decision.stop_playback:
                        await self.player.stop()
                    if decision.interrupt_response and self._responding:
                        await self.session.interrupt()
                    if self.on_event is not None:
                        for generated_event in decision.generated_events:
                            self.on_event(generated_event)
                    if not decision.emit_event:
                        continue
                if (
                    self.command_router is not None
                    and event.type == TRANSCRIPT_DONE
                    and (
                        self.session_controller is None
                        or self.session_controller.allow_playback
                    )
                ):
                    result = await self.command_router.route(event.text)
                    if result is not None:
                        await self.player.stop()
                        if result.should_interrupt_model and self._responding:
                            await self.session.interrupt()
                        if self.on_event is not None:
                            self.on_event(
                                RealtimeEvent(
                                    f"guidebot.command.{result.intent}",
                                    result.response,
                                )
                            )
                        continue
            if self.on_event is not None:
                self.on_event(event)
