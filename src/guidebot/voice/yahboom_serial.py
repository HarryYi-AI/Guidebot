"""Clean-room adapter for the purchased Yahboom offline speech command module."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Mapping


DEFAULT_COMMAND_TEXT: Mapping[int, str] = {
    1: "停止",
    2: "停车",
    3: "休眠",
    4: "前进",
    5: "后退",
    6: "左转",
    7: "右转",
    8: "左旋",
    9: "右旋",
}


@dataclass(frozen=True, slots=True)
class YahboomSpeechEvent:
    language: str
    command_id: int
    text: str | None
    wake_word: bool
    raw_packet: bytes


def parse_yahboom_packet(
    packet: bytes,
    command_text: Mapping[int, str] = DEFAULT_COMMAND_TEXT,
) -> YahboomSpeechEvent:
    """Parse the vendor frame fields used by Speech_Lib v0.0.3.

    The vendor implementation reads byte offsets 2 (language/wake code) and 3
    (command ID). Codes 1..3 select Chinese wake audio and >=4 select English.
    A zero language code denotes a normal command frame.
    """

    if len(packet) < 4:
        raise ValueError("Yahboom speech packet must contain at least four bytes")
    if packet[:2] != b"\xaa\x55":
        raise ValueError("Yahboom speech packet has an invalid header")
    if len(packet) >= 5 and packet[4] != 0xFB:
        raise ValueError("Yahboom speech packet has an invalid terminator")
    language_code = packet[2]
    command_id = packet[3]
    wake_word = language_code != 0
    language = "zh-CN" if language_code < 4 else "en-US"
    text = None if wake_word else command_text.get(command_id)
    return YahboomSpeechEvent(language, command_id, text, wake_word, packet)


class YahboomSerialCommandSource:
    """Reads keyword IDs without importing or modifying the vendor Speech_Lib."""

    def __init__(
        self,
        device: str = "/dev/myspeech",
        baudrate: int = 115_200,
        *,
        serial_port: Any | None = None,
        command_text: Mapping[int, str] = DEFAULT_COMMAND_TEXT,
    ) -> None:
        if serial_port is None:
            try:
                import serial
            except ImportError as error:
                raise RuntimeError(
                    "Yahboom serial input requires the optional 'pyserial' package"
                ) from error
            serial_port = serial.Serial(device, baudrate, timeout=0.2)
        self._port = serial_port
        self.command_text = command_text
        self._buffer = bytearray()

    async def read_event(self) -> YahboomSpeechEvent | None:
        packet = await asyncio.to_thread(self._read_packet)
        return parse_yahboom_packet(packet, self.command_text) if packet else None

    async def read_text(self) -> str | None:
        event = await self.read_event()
        return event.text if event is not None and not event.wake_word else None

    def close(self) -> None:
        close = getattr(self._port, "close", None)
        if close is not None:
            close()

    def _read_packet(self) -> bytes:
        while True:
            header_index = self._buffer.find(b"\xaa\x55")
            if header_index > 0:
                del self._buffer[:header_index]
            elif header_index < 0 and len(self._buffer) > 1:
                del self._buffer[:-1]

            if len(self._buffer) >= 5:
                packet = bytes(self._buffer[:5])
                if packet[4] == 0xFB:
                    del self._buffer[:5]
                    return packet
                del self._buffer[0]
                continue

            chunk = self._port.read(5 - len(self._buffer))
            if not chunk:
                return b""
            self._buffer.extend(chunk)
