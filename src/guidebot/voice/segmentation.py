"""Low-latency semantic segmentation for streaming LLM tokens."""

from __future__ import annotations


class TextSegmenter:
    """Turns arbitrary token fragments into speakable phrases.

    Punctuation closes a phrase once ``min_chars`` is reached. ``max_chars`` is
    a latency bound for models that generate long clauses without punctuation.
    """

    PUNCTUATION = frozenset("。！？；.!?;\n")

    def __init__(self, min_chars: int = 6, max_chars: int = 48) -> None:
        if min_chars < 1 or max_chars < min_chars:
            raise ValueError("segment lengths must satisfy 1 <= min_chars <= max_chars")
        self.min_chars = min_chars
        self.max_chars = max_chars
        self._buffer = ""

    def feed(self, token: str) -> tuple[str, ...]:
        segments: list[str] = []
        for character in token:
            self._buffer += character
            enough = len(self._buffer.strip()) >= self.min_chars
            if (character in self.PUNCTUATION and enough) or len(self._buffer) >= self.max_chars:
                segment = self._take()
                if segment:
                    segments.append(segment)
        return tuple(segments)

    def flush(self) -> str | None:
        segment = self._take()
        return segment or None

    def _take(self) -> str:
        segment = self._buffer
        self._buffer = ""
        return segment

