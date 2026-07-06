from __future__ import annotations

from collections import deque

import pytest

from guidebot.voice.yahboom_serial import (
    YahboomSerialCommandSource,
    parse_yahboom_packet,
)


class FakeSerial:
    def __init__(self, packets: tuple[bytes, ...]) -> None:
        self.packets = deque(packets)
        self.closed = False

    def read(self, size: int) -> bytes:
        return self.packets.popleft() if self.packets else b""

    def close(self) -> None:
        self.closed = True


def test_vendor_packet_parser_separates_wake_and_command_frames() -> None:
    command = parse_yahboom_packet(bytes((0xAA, 0x55, 0, 4, 0xFB)))
    wake = parse_yahboom_packet(bytes((0xAA, 0x55, 1, 0, 0xFB)))

    assert command.text == "前进"
    assert not command.wake_word
    assert wake.wake_word
    assert wake.language == "zh-CN"
    assert wake.text is None


def test_vendor_packet_parser_rejects_truncated_data() -> None:
    with pytest.raises(ValueError, match="at least four"):
        parse_yahboom_packet(b"\xaa\x55")


async def test_serial_source_is_injectable_and_does_not_need_vendor_library() -> None:
    serial = FakeSerial((bytes((0xAA, 0x55, 0, 1, 0xFB)),))
    source = YahboomSerialCommandSource(serial_port=serial)

    assert await source.read_text() == "停止"
    source.close()
    assert serial.closed


async def test_serial_source_accumulates_partial_usb_reads() -> None:
    serial = FakeSerial((b"\xaa", b"\x55\x00", b"\x04", b"\xfb"))
    source = YahboomSerialCommandSource(serial_port=serial)

    assert await source.read_text() == "前进"


async def test_serial_source_resynchronizes_after_noise() -> None:
    serial = FakeSerial((b"\x00\xff", b"\xaa\x55\x00\x05\xfb"))
    source = YahboomSerialCommandSource(serial_port=serial)

    assert await source.read_text() == "后退"
