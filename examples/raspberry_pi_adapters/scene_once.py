#!/usr/bin/env python3
"""Capture one camera frame, call the existing scene VLM code, print JSON.

This wrapper assumes the Yahboom project directory exists at /home/pi/project_demo.
It does not copy or modify the original course code.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
from pathlib import Path


def main() -> None:
    project_demo = Path(os.getenv("PROJECT_DEMO", "/home/pi/project_demo"))
    ai_root = project_demo / "09.AI_Big_Model"
    scene_dir = ai_root / "SceneDescription"
    image_path = scene_dir / "rec.jpg"

    sys.path.insert(0, str(ai_root))
    sys.path.insert(0, str(scene_dir))
    os.chdir(ai_root)

    try:
        import cv2
        from tongyi_api_image import tongyi_Image
    except Exception as exc:  # noqa: BLE001 - adapter must report hardware/env errors.
        emit_event("source.error", {"error": f"scene import failed: {exc}"})
        return

    cap = cv2.VideoCapture(int(os.getenv("GUIDEBOT_CAMERA_INDEX", "0")))
    try:
        cap.set(3, int(os.getenv("GUIDEBOT_CAMERA_WIDTH", "640")))
        cap.set(4, int(os.getenv("GUIDEBOT_CAMERA_HEIGHT", "480")))
        ok, frame = cap.read()
        if not ok:
            emit_event("source.error", {"error": "camera frame capture failed"})
            return
        image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(image_path), frame)
    finally:
        cap.release()

    prompt = (
        "请用一句中文总结画面。重点判断是否存在明火、烟雾、人员摔倒、危险障碍。"
        "如果没有危险，请说明正常。"
    )
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            summary = str(tongyi_Image(prompt))
    except Exception as exc:  # noqa: BLE001
        emit_event("source.error", {"error": f"scene model failed: {exc}"})
        return

    label = classify_scene(summary)
    print(
        json.dumps(
            {"label": label, "summary": summary, "confidence": 0.9},
            ensure_ascii=False,
        ),
        flush=True,
    )


def classify_scene(summary: str) -> str:
    text = summary.casefold()
    if any(word in text for word in ("明火", "火焰", "着火", "起火", "fire", "smoke", "烟")):
        return "fire"
    if any(word in text for word in ("摔倒", "倒地", "跌倒", "fall")):
        return "fall"
    return "normal"


def emit_event(event_type: str, payload: dict[str, object]) -> None:
    print(
        json.dumps(
            {
                "event_type": event_type,
                "source": "scene_once",
                "payload": payload,
                "confidence": 0.0,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
