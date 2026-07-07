from __future__ import annotations

from guidebot.voice.barge_in import BargeInMode, BargeInPolicy
from guidebot.voice.providers import RealtimeEvent


def test_transcript_barge_in_ignores_raw_speech_started() -> None:
    policy = BargeInPolicy(BargeInMode.TRANSCRIPT)

    assert (
        policy.should_interrupt(
            RealtimeEvent("input_audio_buffer.speech_started"),
            responding=True,
        )
        is False
    )


def test_transcript_barge_in_requires_meaningful_text() -> None:
    policy = BargeInPolicy(BargeInMode.TRANSCRIPT, min_transcript_chars=4)

    assert (
        policy.should_interrupt(
            RealtimeEvent("conversation.item.input_audio_transcription.delta", "嗯"),
            responding=True,
        )
        is False
    )
    assert (
        policy.should_interrupt(
            RealtimeEvent(
                "conversation.item.input_audio_transcription.delta",
                "我想问另一个问题",
            ),
            responding=True,
        )
        is True
    )


def test_interrupt_phrases_can_be_short() -> None:
    policy = BargeInPolicy(BargeInMode.TRANSCRIPT, min_transcript_chars=8)

    assert (
        policy.should_interrupt(
            RealtimeEvent("conversation.item.input_audio_transcription.delta", "停一下"),
            responding=True,
        )
        is True
    )


def test_vad_mode_keeps_legacy_immediate_interrupt() -> None:
    policy = BargeInPolicy(BargeInMode.VAD)

    assert (
        policy.should_interrupt(
            RealtimeEvent("input_audio_buffer.speech_started"),
            responding=True,
        )
        is True
    )


def test_off_mode_never_interrupts() -> None:
    policy = BargeInPolicy(BargeInMode.OFF)

    assert (
        policy.should_interrupt(
            RealtimeEvent("conversation.item.input_audio_transcription.delta", "停一下"),
            responding=True,
        )
        is False
    )
