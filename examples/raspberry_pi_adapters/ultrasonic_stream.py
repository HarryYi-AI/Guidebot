#!/usr/bin/env python3
"""Continuously read Yahboom ultrasonic distance and print JSON lines."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def main() -> None:
    project_demo = Path(os.getenv("PROJECT_DEMO", "/home/pi/project_demo"))
    sys.path.insert(0, str(project_demo / "raspbot"))

    from Raspbot_Lib import Raspbot

    interval = float(os.getenv("GUIDEBOT_ULTRASONIC_INTERVAL", "0.05"))
    threshold_mm = int(os.getenv("GUIDEBOT_OBSTACLE_MM", "200"))
    bot = Raspbot()
    bot.Ctrl_Ulatist_Switch(1)
    try:
        while True:
            high = bot.read_data_array(0x1B, 1)[0]
            low = bot.read_data_array(0x1A, 1)[0]
            distance_mm = (high << 8) | low
            print(
                json.dumps(
                    {
                        "obstacle": 0 < distance_mm <= threshold_mm,
                        "distance_mm": distance_mm,
                        "confidence": 1.0,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            time.sleep(interval)
    finally:
        bot.Ctrl_Ulatist_Switch(0)


if __name__ == "__main__":
    main()
