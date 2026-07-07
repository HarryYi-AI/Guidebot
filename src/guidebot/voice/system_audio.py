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
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        if self._process is not None and self._process.returncode is None:
            return
        self._process = None
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
            stderr=asyncio.subprocess.DEVNULL,
        )

    async def play(self, audio: bytes) -> None:
        async with self._lock:
            await self.start()
            process = self._process
            if process is None or process.stdin is None:
                return
            try:
                process.stdin.write(audio)
                await process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                await self._discard_process()

    async def stop(self) -> None:
        async with self._lock:
            await self._discard_process()

    async def _discard_process(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.stdin is not None and not process.stdin.is_closing():
            process.stdin.close()
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
