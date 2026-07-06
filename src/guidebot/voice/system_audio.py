"""ALSA command adapters for Raspberry Pi without Python audio dependencies."""

from __future__ import annotations

import asyncio

from .config import VoiceConfig
from .models import AudioFrame


class ArecordAudioSource:
    """Streams raw PCM from the system ``arecord`` command."""

    def __init__(self, config: VoiceConfig | None = None, device: str | None = None) -> None:
        self.config = config or VoiceConfig()
        self.device = device
        self._process: asyncio.subprocess.Process | None = None

    async def start(self) -> None:
        if self._process is not None:
            return
        args = [
            "arecord",
            "-q",
            "-t",
            "raw",
            "-f",
            "S16_LE",
            "-r",
            str(self.config.sample_rate),
            "-c",
            str(self.config.channels),
        ]
        if self.device:
            args.extend(("-D", self.device))
        self._process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    async def read(self) -> AudioFrame | None:
        await self.start()
        process = self._process
        if process is None or process.stdout is None:
            return None
        target = self.config.frame_bytes
        data = bytearray()
        while len(data) < target:
            chunk = await process.stdout.read(target - len(data))
            if not chunk:
                return None
            data.extend(chunk)
        return AudioFrame(
            bytes(data),
            self.config.sample_rate,
            self.config.channels,
            self.config.sample_width_bytes,
            self.config.chunk_ms,
        )

    async def close(self) -> None:
        if self._process is not None and self._process.returncode is None:
            self._process.terminate()
            await self._process.wait()
        self._process = None


class AplayAudioPlayer:
    """Keeps one raw PCM ``aplay`` process open for low-overhead chunk playback."""

    def __init__(self, config: VoiceConfig | None = None, device: str | None = None) -> None:
        self.config = config or VoiceConfig()
        self.device = device
        self._process: asyncio.subprocess.Process | None = None

    async def start(self) -> None:
        if self._process is not None:
            return
        args = [
            "aplay",
            "-q",
            "-t",
            "raw",
            "-f",
            "S16_LE",
            "-r",
            str(self.config.sample_rate),
            "-c",
            str(self.config.channels),
        ]
        if self.device:
            args.extend(("-D", self.device))
        self._process = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    async def play(self, audio: bytes) -> None:
        await self.start()
        process = self._process
        if process is None or process.stdin is None:
            raise RuntimeError("aplay process has no stdin")
        process.stdin.write(audio)
        await process.stdin.drain()

    async def stop(self) -> None:
        if self._process is not None and self._process.returncode is None:
            self._process.terminate()
            await self._process.wait()
        self._process = None

