import math
import platform
import random
import shutil
import subprocess
import time
from collections import deque
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import yaml


DEFAULT_CONFIG = {
    "camera": {
        "camera_id": 0,
        "frame_width": 640,
        "frame_height": 480,
        "show_window": True,
        "fourcc": "MJPG",
    },
    "face": {
        "max_num_faces": 1,
        "min_detection_confidence": 0.5,
        "min_tracking_confidence": 0.5,
        "refine_landmarks": True,
    },
    "fatigue": {
        "detect_interval": 0.1,
        "window_seconds": 60,
        "eye_closed_ear": 0.21,
        "eye_closed_seconds": 1.5,
        "perclos_threshold": 0.35,
        "yawn_mar": 0.65,
        "yawn_seconds": 1.0,
        "head_down_pitch": 18.0,
        "head_down_seconds": 2.0,
        "fatigue_score_threshold": 3,
        "remind_cooldown_seconds": 60,
    },
    "voice": {
        "mode": "audio_file",
        "audio_file": "sounds/tiredremind.wav",
        "max_seconds": 20,
        "remind_texts": [
            "主人，你看起来很疲劳，早点去休息吧",
        ],
    },
}


LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
MOUTH = (78, 13, 14, 308)
HEAD_POSE_POINTS = (1, 152, 33, 263, 61, 291)


def merge_config(defaults, overrides):
    cfg = deepcopy(defaults)
    if not overrides:
        return cfg

    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key] = merge_config(cfg[key], value)
        else:
            cfg[key] = value
    return cfg


def load_config(path="detface_config.yaml"):
    config_path = Path(path)
    if not config_path.exists():
        config_path = Path(__file__).resolve().parent / path
    if not config_path.exists() or config_path.stat().st_size == 0:
        return deepcopy(DEFAULT_CONFIG)

    with config_path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    return merge_config(DEFAULT_CONFIG, loaded)


def configure_camera(cap, camera_cfg):
    fourcc = camera_cfg.get("fourcc", "MJPG").upper()
    if fourcc and len(fourcc) == 4:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_cfg.get("frame_width", 640))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_cfg.get("frame_height", 480))


def play_audio_file(audio_path, max_seconds):
    if not audio_path.exists():
        print(f"[ERROR] Audio file not found: {audio_path}")
        return

    if platform.system() == "Windows" and audio_path.suffix.lower() == ".wav":
        import winsound  # pylint: disable=import-outside-toplevel

        winsound.PlaySound(str(audio_path), winsound.SND_FILENAME)
        return

    commands = [
        ["aplay", str(audio_path)],
        ["paplay", str(audio_path)],
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(audio_path)],
    ]
    for command in commands:
        if shutil.which(command[0]):
            subprocess.run(command, check=False, timeout=max_seconds)
            return

    print("[ERROR] No supported audio player found. Install aplay, paplay, or ffplay to play reminders.")


def speak(text, voice_cfg):
    mode = voice_cfg.get("mode", "print")
    max_seconds = voice_cfg.get("max_seconds", 20)
    print(f"[REMIND] {text}")

    if mode == "print":
        return

    if mode == "audio_file":
        audio_file = voice_cfg.get("audio_file", "")
        if not audio_file:
            return
        audio_path = Path(audio_file)
        if not audio_path.is_absolute():
            audio_path = Path(__file__).resolve().parent / audio_path
        try:
            play_audio_file(audio_path, max_seconds)
        except subprocess.TimeoutExpired:
            print("[SYSTEM] Audio reminder timeout. Resume detection.")
        except Exception as exc:
            print(f"[ERROR] Failed to play audio file: {exc}")
        return

    if mode == "system_tts":
        try:
            if platform.system() == "Windows":
                script = (
                    "Add-Type -AssemblyName System.Speech; "
                    "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                    "$speaker.Speak($args[0])"
                )
                subprocess.run(
                    ["powershell", "-NoProfile", "-Command", script, text],
                    check=False,
                    timeout=max_seconds,
                )
            else:
                subprocess.run(["espeak-ng", text], check=False, timeout=max_seconds)
        except subprocess.TimeoutExpired:
            print("[SYSTEM] Voice reminder timeout. Resume detection.")
        except Exception as exc:
            print(f"[ERROR] Failed to run system TTS: {exc}")
        return

    if mode == "espeak":
        try:
            subprocess.run(["espeak-ng", "-v", "zh", text], check=False, timeout=max_seconds)
        except subprocess.TimeoutExpired:
            print("[SYSTEM] Voice reminder timeout. Resume detection.")
        except Exception as exc:
            print(f"[ERROR] Failed to run espeak: {exc}")


def landmark_points(face_landmarks, width, height):
    return np.array(
        [(lm.x * width, lm.y * height, lm.z * width) for lm in face_landmarks.landmark],
        dtype=np.float32,
    )


def distance_2d(a, b):
    return float(np.linalg.norm(a[:2] - b[:2]))


def eye_aspect_ratio(points, eye_indices):
    p1, p2, p3, p4, p5, p6 = [points[i] for i in eye_indices]
    vertical = distance_2d(p2, p6) + distance_2d(p3, p5)
    horizontal = 2.0 * distance_2d(p1, p4)
    if horizontal < 1e-6:
        return 0.0
    return vertical / horizontal


def mouth_aspect_ratio(points):
    left, upper, lower, right = [points[i] for i in MOUTH]
    horizontal = distance_2d(left, right)
    if horizontal < 1e-6:
        return 0.0
    return distance_2d(upper, lower) / horizontal


def estimate_head_pitch(points, image_width, image_height):
    image_points = np.array([points[i][:2] for i in HEAD_POSE_POINTS], dtype=np.float64)
    model_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (0.0, -63.6, -12.5),
            (-43.3, 32.7, -26.0),
            (43.3, 32.7, -26.0),
            (-28.9, -28.9, -24.1),
            (28.9, -28.9, -24.1),
        ],
        dtype=np.float64,
    )

    focal_length = image_width
    camera_matrix = np.array(
        [
            [focal_length, 0, image_width / 2],
            [0, focal_length, image_height / 2],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rotation_vec, _ = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return 0.0

    rotation_matrix, _ = cv2.Rodrigues(rotation_vec)
    projection = np.hstack((rotation_matrix, np.zeros((3, 1))))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(projection)
    return float(euler_angles[0][0])


def draw_landmark_group(frame, points, indices, color):
    for idx in indices:
        point = tuple(points[idx][:2].astype(int))
        cv2.circle(frame, point, 2, color, -1)


def draw_status(frame, status):
    color = (0, 0, 255) if status["fatigue"] else (0, 255, 0)
    lines = [
        f"state: {'FATIGUE' if status['fatigue'] else 'normal'}",
        f"score: {status['score']}",
        f"EAR: {status['ear']:.2f}  PERCLOS: {status['perclos']:.2f}",
        f"MAR: {status['mar']:.2f}  pitch: {status['pitch']:.1f}",
        "F: pause/resume  Q: quit",
    ]

    y = 30
    for line in lines:
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        y += 28


def trim_history(history, now, window_seconds):
    while history and now - history[0][0] > window_seconds:
        history.popleft()


def bool_ratio(history):
    if not history:
        return 0.0
    return sum(1 for _, value in history if value) / len(history)


def update_fatigue_state(points, frame_shape, state, fatigue_cfg):
    height, width = frame_shape[:2]
    now = time.time()

    left_ear = eye_aspect_ratio(points, LEFT_EYE)
    right_ear = eye_aspect_ratio(points, RIGHT_EYE)
    ear = (left_ear + right_ear) / 2.0
    mar = mouth_aspect_ratio(points)
    pitch = estimate_head_pitch(points, width, height)

    eye_closed = ear < fatigue_cfg.get("eye_closed_ear", 0.21)
    yawning = mar > fatigue_cfg.get("yawn_mar", 0.65)
    head_down = pitch > fatigue_cfg.get("head_down_pitch", 18.0)

    if eye_closed and state["eye_closed_since"] is None:
        state["eye_closed_since"] = now
    if not eye_closed:
        state["eye_closed_since"] = None

    if yawning and state["yawn_since"] is None:
        state["yawn_since"] = now
    if not yawning:
        state["yawn_since"] = None

    if head_down and state["head_down_since"] is None:
        state["head_down_since"] = now
    if not head_down:
        state["head_down_since"] = None

    state["eye_history"].append((now, eye_closed))
    trim_history(state["eye_history"], now, fatigue_cfg.get("window_seconds", 60))
    perclos = bool_ratio(state["eye_history"])

    prolonged_eye_closed = (
        state["eye_closed_since"] is not None
        and now - state["eye_closed_since"] >= fatigue_cfg.get("eye_closed_seconds", 1.5)
    )
    prolonged_yawn = (
        state["yawn_since"] is not None
        and now - state["yawn_since"] >= fatigue_cfg.get("yawn_seconds", 1.0)
    )
    prolonged_head_down = (
        state["head_down_since"] is not None
        and now - state["head_down_since"] >= fatigue_cfg.get("head_down_seconds", 2.0)
    )
    high_perclos = perclos >= fatigue_cfg.get("perclos_threshold", 0.35)

    score = 0
    score += 2 if prolonged_eye_closed else 0
    score += 2 if high_perclos else 0
    score += 1 if prolonged_yawn else 0
    score += 1 if prolonged_head_down else 0

    fatigue = score >= fatigue_cfg.get("fatigue_score_threshold", 3)
    return {
        "fatigue": fatigue,
        "score": score,
        "ear": ear,
        "mar": mar,
        "pitch": pitch,
        "eye_closed": eye_closed,
        "yawning": yawning,
        "head_down": head_down,
        "perclos": perclos,
    }


def new_state():
    return {
        "eye_closed_since": None,
        "yawn_since": None,
        "head_down_since": None,
        "eye_history": deque(),
    }


def main():
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("mediapipe is required. Install it with: python -m pip install mediapipe") from exc

    cfg = load_config()
    camera_cfg = cfg["camera"]
    face_cfg = cfg["face"]
    fatigue_cfg = cfg["fatigue"]
    voice_cfg = cfg["voice"]

    cap = cv2.VideoCapture(camera_cfg.get("camera_id", 0))
    configure_camera(cap, camera_cfg)
    if not cap.isOpened():
        print("[ERROR] Failed to open camera. Check camera_id or camera permission.")
        return

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=face_cfg.get("max_num_faces", 1),
        refine_landmarks=face_cfg.get("refine_landmarks", True),
        min_detection_confidence=face_cfg.get("min_detection_confidence", 0.5),
        min_tracking_confidence=face_cfg.get("min_tracking_confidence", 0.5),
    )

    state = new_state()
    monitor_enabled = True
    last_detect_time = 0
    last_remind_time = 0
    status = {
        "fatigue": False,
        "score": 0,
        "ear": 0.0,
        "mar": 0.0,
        "pitch": 0.0,
        "perclos": 0.0,
    }

    print("[SYSTEM] Fatigue detector started. Press F to pause/resume, Q to quit.")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read camera frame.")
                break

            now = time.time()
            if monitor_enabled and now - last_detect_time >= fatigue_cfg.get("detect_interval", 0.1):
                last_detect_time = now
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)

                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    points = landmark_points(face_landmarks, frame.shape[1], frame.shape[0])
                    status = update_fatigue_state(points, frame.shape, state, fatigue_cfg)

                    draw_landmark_group(frame, points, LEFT_EYE + RIGHT_EYE, (0, 255, 255))
                    draw_landmark_group(frame, points, MOUTH, (255, 0, 255))

                    cooldown = fatigue_cfg.get("remind_cooldown_seconds", 60)
                    if not status["fatigue"]:
                        last_remind_time = 0
                    if status["fatigue"] and (last_remind_time == 0 or now - last_remind_time >= cooldown):
                        texts = voice_cfg.get("remind_texts", [])
                        text = random.choice(texts) if texts else "Fatigue detected. Please take a break."
                        speak(text, voice_cfg)
                        last_remind_time = now
                else:
                    status = {
                        "fatigue": False,
                        "score": 0,
                        "ear": 0.0,
                        "mar": 0.0,
                        "pitch": 0.0,
                        "perclos": bool_ratio(state["eye_history"]),
                    }

            draw_status(frame, status)

            if camera_cfg.get("show_window", True):
                cv2.imshow("AI Car Fatigue Detector", frame)
                action = cv2.waitKey(1) & 0xFF
                if action in (ord("q"), ord("Q")):
                    break
                if action in (ord("f"), ord("F")):
                    monitor_enabled = not monitor_enabled
                    print(f"[SYSTEM] Monitor {'resumed' if monitor_enabled else 'paused'}.")

            time.sleep(0.001)
    finally:
        face_mesh.close()
        cap.release()
        cv2.destroyAllWindows()
        print("[SYSTEM] Program finished.")


if __name__ == "__main__":
    main()
