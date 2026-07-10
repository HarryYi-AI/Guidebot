import argparse
import base64
import importlib.util
import json
import mimetypes
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
CAPTURE_DIR = BASE_DIR / "captures"
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "monitor_log.jsonl"
ROOT_CONFIG_FILE = PROJECT_ROOT / "API_KEY.py"
LOCAL_CONFIG_FILE = BASE_DIR / "local_config.py"
DEFAULT_WATCH_RULE_FILE = BASE_DIR / "watch_rule.txt"
DEFAULT_MAX_CAPTURE_GB = 5.0

sys.path.append(str(PROJECT_ROOT))


DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_VISION_MODEL = "qwen-vl-plus"

DEFAULT_SCENE_PROMPT = """
你是一个环境观察助手。请根据图片客观描述当前场景，重点关注：
1. 周围有哪些人、物体、设备和明显行为；
2. 是否存在烟雾、火焰、液体泄漏、摔倒、冲突、遮挡通道、门窗异常、电器异常等可见风险；
3. 重点观察门窗是否关闭，尤其说明窗户是否打开、半开、破损、玻璃碎裂或无法判断；
4. 如果图片模糊、过暗、遮挡严重，请明确说明不确定。
请用简洁中文输出，不要编造图片中看不见的信息。
""".strip()

DEFAULT_WATCH_RULE = """
默认把以下情况视为异常或需要提醒：
- 明显火焰、烟雾、漏水、破损、杂物堵塞通道；
- 有人摔倒、长时间躺地、打斗、求助、危险攀爬；
- 门窗异常打开、陌生人靠近重要区域、设备状态明显异常；
- 环境过暗、镜头被遮挡、画面严重模糊导致无法判断。
普通人在场、普通家具物品、正常走动不算异常。
""".strip()


def now_text():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_python_config(file_path):
    spec = importlib.util.spec_from_file_location("auto_scene_monitor_local_config", file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"配置文件加载失败：{file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def is_empty_key(api_key):
    value = str(api_key or "").strip()
    return value == "" or value.lower() in {"xxxxx", "xxxxxxxx", "your_api_key_here"}


_runtime_config = None


def apply_config_file(config, file_path):
    if not file_path.exists():
        return False

    module = load_python_config(file_path)
    api_key = getattr(module, "TONYI_key", "") or getattr(module, "DASHSCOPE_API_KEY", "")
    if is_empty_key(config["tonyi_key"]) and not is_empty_key(api_key):
        config["tonyi_key"] = api_key
        config["config_file"] = str(file_path)

    if hasattr(module, "TTS_IAT_Tongyi"):
        config["tts_iat_tongyi"] = getattr(module, "TTS_IAT_Tongyi")
    if hasattr(module, "TONYI_API_TTS_MODEL"):
        config["tts_model"] = getattr(module, "TONYI_API_TTS_MODEL")
    if hasattr(module, "TONYI_TTS_VOICE"):
        config["tts_voice"] = getattr(module, "TONYI_TTS_VOICE")
    return True


def load_runtime_config():
    global _runtime_config
    if _runtime_config is not None:
        return _runtime_config

    config = {
        "tonyi_key": "",
        "tts_iat_tongyi": True,
        "tts_model": "qwen-tts",
        "tts_voice": "Cherry",
        "config_file": "",
    }

    # Prefer the config shipped with this folder. API_KEY.py remains a fallback
    # so the folder can also reuse the original course configuration.
    for config_file in (LOCAL_CONFIG_FILE, ROOT_CONFIG_FILE):
        try:
            apply_config_file(config, config_file)
        except Exception:
            continue

    _runtime_config = config
    return config


def get_tonyi_key():
    api_key = str(load_runtime_config()["tonyi_key"]).strip()
    if is_empty_key(api_key):
        raise RuntimeError("请先在项目根目录 API_KEY.py 或 AutoSceneMonitor/local_config.py 中填写 TONYI_key。")
    return api_key


def ensure_runtime_dirs():
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_capture_files():
    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [
        path
        for path in CAPTURE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in image_suffixes
    ]


def cleanup_captures(max_gb=DEFAULT_MAX_CAPTURE_GB, protected_paths=None):
    if max_gb <= 0:
        return {"enabled": False, "total_bytes": 0, "deleted": []}

    protected_paths = {Path(path).resolve() for path in protected_paths or []}
    max_bytes = int(max_gb * 1024 * 1024 * 1024)
    files = get_capture_files()
    file_records = []
    total_bytes = 0

    for path in files:
        try:
            stat = path.stat()
        except OSError:
            continue
        total_bytes += stat.st_size
        file_records.append((stat.st_mtime, stat.st_size, path))

    deleted = []
    if total_bytes <= max_bytes:
        return {"enabled": True, "total_bytes": total_bytes, "deleted": deleted}

    for _, size, path in sorted(file_records):
        if path.resolve() in protected_paths:
            continue

        try:
            path.unlink()
        except OSError as exc:
            deleted.append({"path": str(path), "size": size, "error": str(exc)})
            continue

        total_bytes -= size
        deleted.append({"path": str(path), "size": size})
        if total_bytes <= max_bytes:
            break

    return {"enabled": True, "total_bytes": total_bytes, "deleted": deleted}


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_dashscope_client():
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError("缺少原项目视觉模型依赖 openai，请先确认原来的 SceneDescription 环境可以运行。") from exc

    return OpenAI(api_key=get_tonyi_key(), base_url=DASHSCOPE_BASE_URL)


def capture_image(camera_index=0, width=640, height=480, warmup_frames=5):
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError("缺少原项目摄像头依赖 cv2，请先确认原来的 SceneDescription 环境可以运行。") from exc

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"摄像头打开失败：camera_index={camera_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    frame = None
    ok = False
    try:
        for _ in range(max(1, warmup_frames)):
            ok, frame = cap.read()
            time.sleep(0.05)
    finally:
        cap.release()

    if not ok or frame is None:
        raise RuntimeError("摄像头拍照失败：没有读取到有效画面")

    filename = datetime.now().strftime("scene_%Y%m%d_%H%M%S.jpg")
    image_path = CAPTURE_DIR / filename
    if not cv2.imwrite(str(image_path), frame):
        raise RuntimeError(f"图片保存失败：{image_path}")

    return image_path


def describe_scene(image_path, prompt=DEFAULT_SCENE_PROMPT, model=DEFAULT_VISION_MODEL):
    mime_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
    base64_image = encode_image(image_path)
    client = get_dashscope_client()
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}"
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    return completion.choices[0].message.content.strip()


def build_judge_prompt(scene_description, watch_rule=DEFAULT_WATCH_RULE):
    return f"""
你是一个环境异常判断智能体。请根据下面的场景描述判断当前环境是否异常。

判断规则：
{watch_rule}

场景描述：
{scene_description}

请只输出 JSON，不要输出 Markdown，不要添加额外解释。JSON 格式如下：
{{
  "abnormal": true,
  "risk_level": "none/low/medium/high",
  "reason": "判断原因",
  "feedback": "需要反馈给用户的一句话"
}}

字段要求：
- abnormal 必须是 true 或 false；
- risk_level 只能是 none、low、medium、high；
- 如果没有异常，feedback 写“当前环境未发现明显异常。”。
""".strip()


def judge_scene(scene_description, watch_rule=DEFAULT_WATCH_RULE, model=DEFAULT_VISION_MODEL):
    client = get_dashscope_client()
    prompt = build_judge_prompt(scene_description, watch_rule)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "你负责环境安全巡检，只根据输入内容做谨慎判断。",
            },
            {"role": "user", "content": prompt},
        ],
    )
    raw_text = completion.choices[0].message.content.strip()
    return parse_judge_result(raw_text)


def parse_judge_result(raw_text):
    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start >= 0 and end > start:
            try:
                result = json.loads(raw_text[start : end + 1])
            except json.JSONDecodeError:
                result = fallback_judge_result(raw_text)
        else:
            result = fallback_judge_result(raw_text)

    abnormal = bool(result.get("abnormal", False))
    risk_level = str(result.get("risk_level", "none")).lower()
    if risk_level not in {"none", "low", "medium", "high"}:
        risk_level = "medium" if abnormal else "none"

    feedback = str(result.get("feedback", "")).strip()
    if not feedback:
        feedback = "发现环境异常，请及时查看。" if abnormal else "当前环境未发现明显异常。"

    return {
        "abnormal": abnormal,
        "risk_level": risk_level,
        "reason": str(result.get("reason", "")).strip(),
        "feedback": feedback,
        "raw": raw_text,
    }


def fallback_judge_result(raw_text):
    danger_words = ["异常", "危险", "火", "烟", "摔倒", "泄漏", "打斗", "遮挡", "破损"]
    abnormal = any(word in raw_text for word in danger_words)
    return {
        "abnormal": abnormal,
        "risk_level": "medium" if abnormal else "none",
        "reason": "模型未返回标准 JSON，已根据原始文本做保守判断。",
        "feedback": raw_text if raw_text else "判断模型未返回有效内容。",
    }


def play_audio_file(audio_path):
    audio_path = str(Path(audio_path))
    players = [
        ("mplayer", ["mplayer", audio_path]),
        ("aplay", ["aplay", audio_path]),
        ("ffplay", ["ffplay", "-nodisp", "-autoexit", audio_path]),
        ("mpg123", ["mpg123", audio_path]),
    ]
    errors = []

    for player_name, command in players:
        if not shutil.which(player_name):
            continue

        completed = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        if completed.returncode == 0:
            return player_name
        errors.append(f"{player_name}: {completed.stderr.strip()}")

    if os.name == "nt":
        try:
            import winsound

            winsound.PlaySound(audio_path, winsound.SND_FILENAME)
            return "winsound"
        except Exception as exc:
            errors.append(f"winsound: {exc}")

    detail = "；".join(error for error in errors if error)
    if detail:
        raise RuntimeError(f"音频文件已生成，但播放失败：{detail}")
    raise RuntimeError("音频文件已生成，但没有找到可用播放器，请确认 mplayer、aplay、ffplay 或 mpg123 可用。")


def tonyi_tts_local(text):
    try:
        import dashscope
        import requests
    except ModuleNotFoundError as exc:
        raise RuntimeError("缺少原项目语音播报依赖 dashscope 或 requests，请先确认原来的 TTS 环境可以运行。") from exc

    config = load_runtime_config()
    print(f"[{now_text()}] 开始生成语音：{text}")
    response = dashscope.audio.qwen_tts.SpeechSynthesizer.call(
        model=config["tts_model"],
        api_key=get_tonyi_key(),
        text=text,
        voice=config["tts_voice"],
    )

    audio_url = response.output.audio["url"]
    save_path = BASE_DIR / "answer.wav"
    audio_response = requests.get(audio_url, timeout=20)
    audio_response.raise_for_status()
    with open(save_path, "wb") as audio_file:
        audio_file.write(audio_response.content)

    print(f"[{now_text()}] 语音文件已保存：{save_path}")
    player_name = play_audio_file(save_path)
    print(f"[{now_text()}] 语音播报完成，播放器：{player_name}")


def speak_feedback(text):
    try:
        if load_runtime_config()["tts_iat_tongyi"]:
            tonyi_tts_local(text)
        else:
            from SceneDescription.xinghou_tts import Xinghou_speaktts

            Xinghou_speaktts(text)
        return True
    except Exception as exc:
        print(f"[{now_text()}] 语音反馈失败：{exc}")
        print(f"[{now_text()}] 请先运行：python3 AutoSceneMonitor/auto_scene_monitor.py --test-speak")
        return False


def write_log(record):
    with open(LOG_FILE, "a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_once(args):
    image_path = capture_image(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        warmup_frames=args.warmup_frames,
    )
    cleanup_result = cleanup_captures(args.max_capture_gb, protected_paths=[image_path])
    scene_description = describe_scene(
        image_path=image_path,
        prompt=args.scene_prompt,
        model=args.vision_model,
    )
    judge_result = judge_scene(
        scene_description=scene_description,
        watch_rule=args.watch_rule,
        model=args.judge_model,
    )

    record = {
        "time": now_text(),
        "image_path": str(image_path),
        "scene_description": scene_description,
        "judge_result": judge_result,
        "capture_cleanup": cleanup_result,
    }
    write_log(record)

    print("\n" + "=" * 60)
    print(f"时间：{record['time']}")
    print(f"图片：{record['image_path']}")
    print(f"场景描述：{scene_description}")
    print(f"异常判断：{json.dumps(judge_result, ensure_ascii=False)}")
    if cleanup_result["deleted"]:
        print(f"照片容量超过限制，已删除 {len(cleanup_result['deleted'])} 张旧照片。")

    if judge_result["abnormal"]:
        print(f"用户反馈：{judge_result['feedback']}")
        if args.speak:
            speak_feedback(judge_result["feedback"])
    elif args.report_normal:
        print(judge_result["feedback"])

    return record


def load_text_file(path, default_text):
    if not path:
        return default_text
    file_path = Path(path)
    if not file_path.exists():
        return default_text
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read().strip()


def run_test_speak(text):
    ensure_runtime_dirs()
    print(f"[{now_text()}] 开始语音播报自检。")
    ok = speak_feedback(text)
    if ok:
        print(f"[{now_text()}] 语音播报自检完成。")
    else:
        print(f"[{now_text()}] 语音播报自检失败，请根据上面的错误检查播放器、网络或音频输出。")


def parse_args():
    parser = argparse.ArgumentParser(description="自动场景巡检：定时拍照、视觉描述、异常判断、用户反馈。")
    parser.add_argument("--interval", type=float, default=300, help="两次自动巡检之间的间隔秒数，默认 300。")
    parser.add_argument("--once", action="store_true", help="只巡检一次，不进入循环。")
    parser.add_argument("--camera", type=int, default=0, help="摄像头编号，默认 0。")
    parser.add_argument("--width", type=int, default=640, help="拍照宽度，默认 640。")
    parser.add_argument("--height", type=int, default=480, help="拍照高度，默认 480。")
    parser.add_argument("--warmup-frames", type=int, default=5, help="拍照前预读帧数，默认 5。")
    parser.add_argument("--vision-model", default=DEFAULT_VISION_MODEL, help="视觉大模型名称。")
    parser.add_argument("--judge-model", default="", help="异常判断模型名称，默认与视觉大模型相同。")
    parser.add_argument("--scene-prompt-file", default="", help="自定义视觉描述提示词文件。")
    parser.add_argument(
        "--watch-rule-file",
        default=str(DEFAULT_WATCH_RULE_FILE),
        help="自定义异常判断规则文件，默认读取 AutoSceneMonitor/watch_rule.txt。",
    )
    parser.add_argument(
        "--max-capture-gb",
        type=float,
        default=DEFAULT_MAX_CAPTURE_GB,
        help="captures 目录照片总容量上限，超过后删除最旧照片，默认 5GB；设为 0 表示不清理。",
    )
    parser.add_argument(
        "--speak",
        dest="speak",
        action="store_true",
        default=True,
        help="发现异常时调用 TTS 语音播报，默认开启。",
    )
    parser.add_argument(
        "--no-speak",
        dest="speak",
        action="store_false",
        help="关闭异常语音播报，仅在控制台输出反馈。",
    )
    parser.add_argument("--test-speak", action="store_true", help="只测试语音播报，不拍照、不调用视觉模型。")
    parser.add_argument(
        "--test-speak-text",
        default="语音播报测试，当前场景巡检系统可以正常播报。",
        help="语音播报自检文本。",
    )
    parser.add_argument("--report-normal", action="store_true", help="无异常时也在控制台输出反馈句。")
    args = parser.parse_args()
    if not args.judge_model:
        args.judge_model = args.vision_model
    args.scene_prompt = load_text_file(args.scene_prompt_file, DEFAULT_SCENE_PROMPT)
    args.watch_rule = load_text_file(args.watch_rule_file, DEFAULT_WATCH_RULE)
    return args


def main():
    ensure_runtime_dirs()
    args = parse_args()
    if args.test_speak:
        run_test_speak(args.test_speak_text)
        return

    print(f"[{now_text()}] 自动场景巡检启动，间隔 {args.interval} 秒。按 Ctrl+C 停止。")
    print(f"[{now_text()}] 视觉描述模型：{args.vision_model}；异常判断模型：{args.judge_model}")

    while True:
        start_time = time.time()
        try:
            run_once(args)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            error_record = {"time": now_text(), "error": str(exc)}
            write_log(error_record)
            print(f"[{now_text()}] 本轮巡检失败：{exc}")

        if args.once:
            break

        elapsed = time.time() - start_time
        time.sleep(max(0, args.interval - elapsed))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n[{now_text()}] 自动场景巡检已停止。")
