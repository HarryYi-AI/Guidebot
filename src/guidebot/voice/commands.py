"""Local voice command routing for non-chat system actions."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Protocol


class VolumeController(Protocol):
    async def set_percent(self, percent: int) -> None: ...

    async def change_percent(self, delta: int) -> None: ...

    async def mute(self) -> None: ...

    async def unmute(self) -> None: ...


@dataclass(frozen=True, slots=True)
class VoiceCommandResult:
    intent: str
    response: str
    should_interrupt_model: bool = True


class AlsaVolumeController:
    """Control Raspberry Pi playback volume through ``amixer``."""

    def __init__(
        self,
        *,
        device: str = "default",
        mixer: str = "Master",
        fallback_mixers: tuple[str, ...] = ("PCM", "Speaker", "Headphone"),
    ) -> None:
        self.device = device
        self.mixers = (mixer, *tuple(name for name in fallback_mixers if name != mixer))

    async def set_percent(self, percent: int) -> None:
        await self._run_for_first_available(f"{_clamp_percent(percent)}%")

    async def change_percent(self, delta: int) -> None:
        suffix = "+" if delta >= 0 else "-"
        await self._run_for_first_available(f"{abs(delta)}%{suffix}")

    async def mute(self) -> None:
        await self._run_for_first_available("mute")

    async def unmute(self) -> None:
        await self._run_for_first_available("unmute")

    async def _run_for_first_available(self, value: str) -> None:
        errors: list[str] = []
        for mixer in self.mixers:
            process = await asyncio.create_subprocess_exec(
                "amixer",
                "-D",
                self.device,
                "sset",
                mixer,
                value,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _stdout, stderr = await process.communicate()
            if process.returncode == 0:
                return
            errors.append(stderr.decode("utf-8", errors="replace").strip())
        raise RuntimeError("failed to set ALSA volume: " + "; ".join(errors))


class VoiceIntentRouter:
    """Routes short user utterances to local skills before generic chat."""

    def __init__(self, volume: VolumeController, *, volume_step: int = 10) -> None:
        if volume_step <= 0:
            raise ValueError("volume_step must be positive")
        self.volume = volume
        self.volume_step = volume_step

    async def route(self, text: str) -> VoiceCommandResult | None:
        normalized = _normalize(text)
        if not _mentions_audio(normalized):
            return None

        if "取消静音" in normalized or "恢复声音" in normalized or "打开声音" in normalized:
            await self.volume.unmute()
            return VoiceCommandResult("volume.unmute", "已取消静音。")

        if "静音" in normalized or "没有声音" in normalized:
            await self.volume.mute()
            return VoiceCommandResult("volume.mute", "已静音。")

        percent = _extract_percent(normalized)
        if percent is not None:
            await self.volume.set_percent(percent)
            return VoiceCommandResult("volume.set", f"已把音量设为 {percent}%。")

        if _contains_any(normalized, ("大一点", "调大", "提高", "增大", "加大", "声音大")):
            await self.volume.change_percent(self.volume_step)
            return VoiceCommandResult("volume.up", "已调大音量。")

        if _contains_any(normalized, ("小一点", "调小", "降低", "减小", "声音小")):
            await self.volume.change_percent(-self.volume_step)
            return VoiceCommandResult("volume.down", "已调小音量。")

        return None


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text.casefold())


def _mentions_audio(text: str) -> bool:
    return "音量" in text or "声音" in text or "静音" in text


def _extract_percent(text: str) -> int | None:
    if "最大" in text or "开满" in text:
        return 100
    if "最小" in text:
        return 10
    match = re.search(r"(?:音量|声音)(?:调到|设为|设置为|到)?(\d{1,3})%?", text)
    if match is None:
        return None
    return _clamp_percent(int(match.group(1)))


def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in text for pattern in patterns)


def _clamp_percent(percent: int) -> int:
    return max(0, min(100, percent))
