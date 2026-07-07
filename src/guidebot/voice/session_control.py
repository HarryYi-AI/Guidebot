"""Wake/sleep state machine for realtime voice sessions."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .providers.dashscope_realtime import RealtimeEvent

TRANSCRIPT_DONE = "conversation.item.input_audio_transcription.completed"


@dataclass(frozen=True, slots=True)
class SessionControlDecision:
    emit_event: bool = True
    interrupt_response: bool = False
    stop_playback: bool = False
    generated_events: tuple[RealtimeEvent, ...] = ()


@dataclass(slots=True)
class WakeSleepController:
    """Controls whether Guidebot is actively conversing or waiting for a wake phrase."""

    wake_phrases: tuple[str, ...] = ("你好guidebot", "你好小盖", "小盖同学")
    sleep_phrases: tuple[str, ...] = (
        "今天聊到这里",
        "结束对话",
        "先这样",
        "不用聊了",
        "休眠",
    )
    require_wake: bool = False
    active: bool | None = None
    debug_inactive_transcripts: bool = False
    _normalized_wake_phrases: tuple[str, ...] = field(init=False, repr=False)
    _normalized_sleep_phrases: tuple[str, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.active = not self.require_wake if self.active is None else self.active
        self._normalized_wake_phrases = tuple(
            normalized for phrase in self.wake_phrases if (normalized := _normalize(phrase))
        )
        self._normalized_sleep_phrases = tuple(
            normalized for phrase in self.sleep_phrases if (normalized := _normalize(phrase))
        )

    @property
    def allow_playback(self) -> bool:
        return bool(self.active)

    def update(self, event: RealtimeEvent) -> SessionControlDecision:
        if event.type != TRANSCRIPT_DONE:
            return SessionControlDecision(emit_event=self.active or not self.require_wake)

        text = _normalize(event.text)
        if self.active:
            if self._matches(text, self._normalized_sleep_phrases):
                self.active = False
                return SessionControlDecision(
                    interrupt_response=True,
                    stop_playback=True,
                    generated_events=(
                        RealtimeEvent("guidebot.session.sleep", "已结束对话，进入待机。"),
                    ),
                )
            return SessionControlDecision()

        if self._matches(text, self._normalized_wake_phrases):
            self.active = True
            return SessionControlDecision(
                stop_playback=True,
                generated_events=(RealtimeEvent("guidebot.session.awake", "我在。"),),
            )

        return SessionControlDecision(
            emit_event=self.debug_inactive_transcripts,
            interrupt_response=True,
            stop_playback=True,
        )

    @staticmethod
    def _matches(text: str, phrases: tuple[str, ...]) -> bool:
        return any(phrase in text for phrase in phrases)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text.casefold())
