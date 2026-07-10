"""Convert input text to a WAV speech file.

This module intentionally keeps the TTS dependency optional. Install edge-tts
when speech generation is needed, and install ffmpeg for WAV conversion:

    python -m pip install edge-tts

Examples:
    python src/guidebot/health_guardian/t2v.py "久坐提醒"
    python src/guidebot/health_guardian/t2v.py "久坐提醒" -o sounds/remind.wav
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import subprocess
import tempfile
from pathlib import Path


DEFAULT_VOICE = "zh-CN-XiaoxiaoNeural"
DEFAULT_OUTPUT = "sounds/tiredremind.wav"


def _resolve_output_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    if path.suffix.lower() != ".wav":
        raise ValueError(f"Output path must end with .wav: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


async def _edge_tts_to_wav(
    text: str,
    output_path: Path,
    *,
    voice: str,
    rate: str,
    volume: str,
) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to generate WAV files. Install ffmpeg and try again.")

    try:
        import edge_tts  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise RuntimeError(
            "edge-tts is required to generate WAV files. "
            "Install it with: python -m pip install edge-tts"
        ) from exc

    temp_mp3 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    temp_mp3_path = Path(temp_mp3.name)
    temp_mp3.close()

    try:
        communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate, volume=volume)
        await communicate.save(str(temp_mp3_path))
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(temp_mp3_path),
                "-acodec",
                "pcm_s16le",
                "-ar",
                "24000",
                "-ac",
                "1",
                str(output_path),
            ],
            check=True,
            timeout=60,
        )
    finally:
        temp_mp3_path.unlink(missing_ok=True)

    return output_path


def text_to_wav(
    text: str,
    output_path: str | Path = DEFAULT_OUTPUT,
    *,
    voice: str = DEFAULT_VOICE,
    rate: str = "+0%",
    volume: str = "+0%",
) -> Path:
    """Convert text to a WAV file and return the generated file path.

    Args:
        text: Text to synthesize. Empty or whitespace-only text is rejected.
        output_path: Target WAV path. Relative paths are resolved beside this file.
        voice: edge-tts voice name, e.g. ``zh-CN-XiaoxiaoNeural``.
        rate: Speech speed accepted by edge-tts, e.g. ``+0%`` or ``-10%``.
        volume: Speech volume accepted by edge-tts, e.g. ``+0%`` or ``+20%``.
    """

    normalized_text = text.strip()
    if not normalized_text:
        raise ValueError("Text cannot be empty.")

    resolved_output = _resolve_output_path(output_path)
    return asyncio.run(
        _edge_tts_to_wav(
            normalized_text,
            resolved_output,
            voice=voice,
            rate=rate,
            volume=volume,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert text to a WAV speech file.")
    parser.add_argument("text", nargs="?", help="Text to synthesize. Reads stdin if omitted.")
    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output WAV path. Defaults to {DEFAULT_OUTPUT}.",
    )
    parser.add_argument(
        "--voice",
        default=DEFAULT_VOICE,
        help=f"edge-tts voice name. Defaults to {DEFAULT_VOICE}.",
    )
    parser.add_argument("--rate", default="+0%", help="Speech rate, e.g. +0%% or -10%%.")
    parser.add_argument("--volume", default="+0%", help="Speech volume, e.g. +0%% or +20%%.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    text = args.text if args.text is not None else input("Text: ")
    output = text_to_wav(
        text,
        args.output,
        voice=args.voice,
        rate=args.rate,
        volume=args.volume,
    )
    print(f"[SYSTEM] WAV generated: {output}")


if __name__ == "__main__":
    main()
