#!/usr/bin/env python3
"""Health detector adapter placeholder.

Replace `detect_health()` with your existing YOLOv8 posture/fatigue call. Keep
the JSON output shape stable so Guidebot can schedule reminders and cooldowns.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def main() -> None:
    result = detect_health()
    print(json.dumps(result, ensure_ascii=False), flush=True)


def detect_health() -> dict[str, object]:
    state_json = os.getenv("GUIDEBOT_HEALTH_STATE_JSON")
    if state_json:
        return json.loads(state_json)

    state_file = Path(os.getenv("GUIDEBOT_HEALTH_STATE_FILE", "/tmp/guidebot_health.json"))
    if state_file.exists():
        return json.loads(state_file.read_text(encoding="utf-8"))

    return {
        "label": "normal",
        "sedentary": False,
        "fatigue": False,
        "confidence": 0.8,
    }


if __name__ == "__main__":
    main()
