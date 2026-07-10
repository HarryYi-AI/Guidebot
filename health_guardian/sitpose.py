import json
import os

import cv2  # 导入 OpenCV，用于摄像头读取、画框、画文字和显示窗口。
import platform
import random  # 导入 random，用于从多条提醒文案中随机选择一句。
import shutil  # 导入 shutil，用于查找系统里可用的音频播放器。
import subprocess  # 导入 subprocess，用于调用系统命令播放音频或 TTS。
import time  # 导入 time，用于计时、计算 FPS 和控制检测间隔。
from collections import deque  # 导入 deque，用固定长度队列保存最近检测结果。
from copy import deepcopy  # 导入 deepcopy，用于复制默认配置，避免修改原始模板。
from pathlib import Path  # 导入 Path，用于更方便地检查配置文件路径。

import numpy as np  # 导入 NumPy，用于关键点坐标和角度计算。
import yaml  # 导入 yaml，用于读取 config.yaml 配置文件。


DEFAULT_CONFIG = {  # 默认配置；config.yaml 为空时程序会使用这一份配置。
    "camera": {  # 摄像头相关配置。
        "camera_id": 0,  # 摄像头编号，0 通常表示默认 USB 摄像头。
        "frame_width": 640,  # 摄像头画面宽度。
        "frame_height": 480,  # 摄像头画面高度。
        "show_window": True,  # 是否显示 OpenCV 实时窗口。
        "fourcc": "MJPG",  # 摄像头视频编码格式，小车示例里常用 MJPG。
    },  # 摄像头配置结束。
    "model": {  # 姿态估计模型相关配置。
        "backend": "ultralytics",  # 推理后端：ultralytics 使用 .pt；opencv_onnx 使用 .onnx。
        "model_path": "yolov8n-pose.pt",  # YOLO Pose 模型路径。
        "imgsz": 320,  # 模型输入尺寸，越小越快但精度可能降低。
        "conf": 0.35,  # 人体检测框置信度阈值。
        "nms": 0.45,  # OpenCV ONNX 后端使用的非极大值抑制阈值。
        "keypoint_conf": 0.25,  # 人体关键点置信度阈值。
    },  # 模型配置结束。
    "roi": {  # ROI 监测区域配置。
        "enabled": False,  # 是否启用 ROI 区域限制。
        "x1": 0,  # ROI 左上角 x 坐标。
        "y1": 0,  # ROI 左上角 y 坐标。
        "x2": 640,  # ROI 右下角 x 坐标。
        "y2": 480,  # ROI 右下角 y 坐标。
    },  # ROI 配置结束。
    "sitting": {  # 久坐检测相关配置。
        "sit_limit_seconds": 60,  # 连续坐姿超过多少秒后提醒。
        "reset_seconds": 30,  # 离开坐姿多少秒后清零计时。
        "remind_cooldown_seconds": 60,  # 两次提醒之间的最小间隔。
        "detect_interval": 0.5,  # 每隔多少秒执行一次姿态检测。
        "smooth_window": 10,  # 用最近多少次检测结果做平滑。
        "sitting_ratio_threshold": 0.6,  # 最近结果中坐姿比例达到多少才认为稳定坐姿。
    },  # 久坐配置结束。
    "voice": {  # 语音或文字提醒相关配置。
        "mode": "audio_file",  # 提醒模式：print、audio_file、system_tts 或 espeak。
        "audio_file": "sounds/sitremind.mp3",  # audio_file 模式下播放的音频文件。
        "max_seconds": 20,  # 单次提醒最多播放多少秒，超时后自动回到检测。
        "remind_texts": [  # 可随机选择的提醒文案列表。
            "主人，你已经长时间坐在椅子上了，该起来活动一下了。",
        ],  # 提醒文案列表结束。
    },  # 提醒配置结束。
}  # 默认配置结束。

COCO_SKELETON = [  # COCO 17 关键点格式下的骨架连线关系。
    (5, 6),  # 左肩连接右肩。
    (5, 7),  # 左肩连接左肘。
    (7, 9),  # 左肘连接左手腕。
    (6, 8),  # 右肩连接右肘。
    (8, 10),  # 右肘连接右手腕。
    (5, 11),  # 左肩连接左髋。
    (6, 12),  # 右肩连接右髋。
    (11, 12),  # 左髋连接右髋。
    (11, 13),  # 左髋连接左膝。
    (13, 15),  # 左膝连接左脚踝。
    (12, 14),  # 右髋连接右膝。
    (14, 16),  # 右膝连接右脚踝。
]  # 骨架连线列表结束。


def merge_config(defaults, overrides):  # 定义递归合并配置的函数。
    cfg = deepcopy(defaults)  # 复制默认配置，避免直接修改 DEFAULT_CONFIG。
    if not overrides:  # 如果用户配置为空。
        return cfg  # 直接返回默认配置副本。

    for key, value in overrides.items():  # 遍历用户配置里的每个键和值。
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):  # 如果默认值和用户值都是字典。
            cfg[key] = merge_config(cfg[key], value)  # 递归合并子配置。
        else:  # 如果不是字典，说明是普通配置项。
            cfg[key] = value  # 用用户配置覆盖默认配置。
    return cfg  # 返回合并后的完整配置。


def load_config(path=None):  # 定义读取配置文件的函数。
    if path is None:
        path = Path(__file__).resolve().parent / "config.yaml"
    config_path = Path(path)  # 把配置文件路径转换成 Path 对象。
    if not config_path.exists() or config_path.stat().st_size == 0:  # 如果配置不存在或为空。
        return deepcopy(DEFAULT_CONFIG)  # 返回默认配置副本。

    with config_path.open("r", encoding="utf-8") as f:  # 用 UTF-8 打开配置文件。
        loaded = yaml.safe_load(f)  # 使用 yaml 解析配置文件内容。

    return merge_config(DEFAULT_CONFIG, loaded)  # 把用户配置合并到默认配置上。


def letterbox(frame, size):  # 定义 YOLO 常用的等比例缩放填充函数。
    height, width = frame.shape[:2]  # 读取原图高度和宽度。
    scale = min(size / width, size / height)  # 计算缩放比例。
    new_width = int(round(width * scale))  # 计算缩放后的宽度。
    new_height = int(round(height * scale))  # 计算缩放后的高度。
    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)  # 缩放图像。

    pad_x = (size - new_width) / 2  # 计算左右填充。
    pad_y = (size - new_height) / 2  # 计算上下填充。
    left = int(round(pad_x - 0.1))  # 左侧填充像素数。
    right = int(round(pad_x + 0.1))  # 右侧填充像素数。
    top = int(round(pad_y - 0.1))  # 顶部填充像素数。
    bottom = int(round(pad_y + 0.1))  # 底部填充像素数。
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))  # 填充图像。
    return padded, scale, left, top  # 返回预处理图像和坐标还原参数。


def load_pose_model(model_cfg):  # 定义姿态模型加载函数，支持 Ultralytics 和 OpenCV ONNX。
    backend = model_cfg.get("backend", "ultralytics").lower()  # 读取推理后端。
    model_path = model_cfg["model_path"]  # 读取模型路径。

    if backend == "ultralytics":  # 如果使用 Ultralytics 后端。
        try:  # 延迟导入，避免树莓派没有 ultralytics 时程序启动就失败。
            from ultralytics import YOLO  # pylint: disable=import-outside-toplevel
        except ImportError as exc:  # 如果没有安装 Ultralytics。
            raise RuntimeError("Ultralytics is not installed. Set model.backend to opencv_onnx or install ultralytics.") from exc
        return backend, YOLO(model_path)  # 返回后端名称和模型对象。

    if backend == "opencv_onnx":  # 如果使用 OpenCV DNN 加载 ONNX 模型。
        net = cv2.dnn.readNetFromONNX(model_path)  # 读取 ONNX 模型。
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)  # 使用 OpenCV 后端。
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # 树莓派上默认使用 CPU。
        return backend, net  # 返回后端名称和网络对象。

    raise ValueError(f"Unsupported model backend: {backend}")  # 不支持的后端直接报错。


def predict_pose(model, backend, frame, model_cfg):  # 定义统一姿态预测函数。
    if backend == "ultralytics":  # Ultralytics 后端保持原来的调用方式。
        results = model.predict(
            frame,
            imgsz=model_cfg.get("imgsz", 320),
            conf=model_cfg.get("conf", 0.35),
            verbose=False,
        )
        result = results[0]
        if result.boxes is None or result.keypoints is None:  # 没有检测结果时返回空数组。
            return np.empty((0, 4), dtype=np.float32), np.empty((0, 17, 2), dtype=np.float32), None

        boxes = result.boxes.xyxy.cpu().numpy()  # 提取人体框坐标。
        kpts_all = result.keypoints.xy.cpu().numpy()  # 提取关键点坐标。
        conf_all = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None  # 提取关键点置信度。
        return boxes, kpts_all, conf_all  # 返回统一格式结果。

    if backend == "opencv_onnx":  # OpenCV ONNX 后端。
        return predict_pose_opencv_onnx(model, frame, model_cfg)  # 调用 ONNX 解码函数。

    raise ValueError(f"Unsupported model backend: {backend}")  # 不支持的后端直接报错。


def predict_pose_opencv_onnx(net, frame, model_cfg):  # 定义 OpenCV DNN YOLOv8-pose ONNX 推理函数。
    imgsz = int(model_cfg.get("imgsz", 320))  # 读取模型输入尺寸。
    conf_threshold = float(model_cfg.get("conf", 0.35))  # 读取检测置信度阈值。
    nms_threshold = float(model_cfg.get("nms", 0.45))  # 读取 NMS 阈值。
    image_height, image_width = frame.shape[:2]  # 读取原始图像尺寸。

    input_image, scale, pad_x, pad_y = letterbox(frame, imgsz)  # 做 YOLO 风格预处理。
    blob = cv2.dnn.blobFromImage(input_image, 1 / 255.0, (imgsz, imgsz), swapRB=True, crop=False)  # 转成模型输入。
    net.setInput(blob)  # 设置网络输入。
    output = net.forward()  # 执行推理。

    predictions = np.squeeze(output)  # 去掉 batch 维度。
    if predictions.ndim != 2:  # 如果输出形状异常。
        return np.empty((0, 4), dtype=np.float32), np.empty((0, 17, 2), dtype=np.float32), None
    if predictions.shape[0] < predictions.shape[1]:  # YOLOv8 ONNX 常见输出为 [56, 2100]，需要转置。
        predictions = predictions.T

    boxes_xyxy = []  # 保存 xyxy 格式框。
    boxes_xywh = []  # 保存 OpenCV NMS 需要的 xywh 框。
    scores = []  # 保存检测分数。
    keypoints = []  # 保存关键点坐标。
    keypoint_scores = []  # 保存关键点置信度。

    for pred in predictions:  # 遍历每个候选框。
        if len(pred) < 56:  # YOLOv8n-pose 应为 4 + 1 + 17 * 3。
            continue
        score = float(pred[4])  # 单类别 pose 模型的 person 置信度。
        if score < conf_threshold:  # 过滤低置信度结果。
            continue

        cx, cy, width, height = pred[:4]  # 读取中心点和宽高。
        x1 = (cx - width / 2 - pad_x) / scale  # 还原左上角 x。
        y1 = (cy - height / 2 - pad_y) / scale  # 还原左上角 y。
        x2 = (cx + width / 2 - pad_x) / scale  # 还原右下角 x。
        y2 = (cy + height / 2 - pad_y) / scale  # 还原右下角 y。
        x1 = float(np.clip(x1, 0, image_width - 1))  # 限制到图像范围。
        y1 = float(np.clip(y1, 0, image_height - 1))  # 限制到图像范围。
        x2 = float(np.clip(x2, 0, image_width - 1))  # 限制到图像范围。
        y2 = float(np.clip(y2, 0, image_height - 1))  # 限制到图像范围。
        if x2 <= x1 or y2 <= y1:  # 跳过异常框。
            continue

        raw_kpts = pred[5:56].reshape(17, 3)  # 拆出 17 个 COCO 关键点。
        points = raw_kpts[:, :2].copy()  # 复制关键点坐标。
        points[:, 0] = (points[:, 0] - pad_x) / scale  # 还原关键点 x。
        points[:, 1] = (points[:, 1] - pad_y) / scale  # 还原关键点 y。
        points[:, 0] = np.clip(points[:, 0], 0, image_width - 1)  # 限制关键点 x。
        points[:, 1] = np.clip(points[:, 1], 0, image_height - 1)  # 限制关键点 y。

        boxes_xyxy.append([x1, y1, x2, y2])  # 保存 xyxy 框。
        boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])  # 保存 xywh 框。
        scores.append(score)  # 保存检测分数。
        keypoints.append(points.astype(np.float32))  # 保存关键点坐标。
        keypoint_scores.append(raw_kpts[:, 2].astype(np.float32))  # 保存关键点置信度。

    if not boxes_xyxy:  # 没有检测结果时返回空数组。
        return np.empty((0, 4), dtype=np.float32), np.empty((0, 17, 2), dtype=np.float32), None

    indices = cv2.dnn.NMSBoxes(boxes_xywh, scores, conf_threshold, nms_threshold)  # 做非极大值抑制。
    indices = np.array(indices).reshape(-1) if len(indices) else []  # 兼容不同 OpenCV 版本的返回格式。
    if len(indices) == 0:  # NMS 后没有结果。
        return np.empty((0, 4), dtype=np.float32), np.empty((0, 17, 2), dtype=np.float32), None

    return (  # 返回统一格式的检测结果。
        np.array([boxes_xyxy[i] for i in indices], dtype=np.float32),
        np.array([keypoints[i] for i in indices], dtype=np.float32),
        np.array([keypoint_scores[i] for i in indices], dtype=np.float32),
    )


def valid_point(p):  # 定义判断关键点是否有效的函数。
    return p is not None and p[0] > 0 and p[1] > 0  # 坐标存在且 x、y 大于 0 就认为有效。


def calc_angle(a, b, c):  # 定义计算三点夹角的函数，角度顶点是 b。
    a = np.array(a, dtype=np.float32)  # 把点 a 转成 NumPy 浮点数组。
    b = np.array(b, dtype=np.float32)  # 把点 b 转成 NumPy 浮点数组。
    c = np.array(c, dtype=np.float32)  # 把点 c 转成 NumPy 浮点数组。

    ba = a - b  # 计算从 b 指向 a 的向量。
    bc = c - b  # 计算从 b 指向 c 的向量。

    norm_ba = np.linalg.norm(ba)  # 计算向量 ba 的长度。
    norm_bc = np.linalg.norm(bc)  # 计算向量 bc 的长度。
    if norm_ba < 1e-6 or norm_bc < 1e-6:  # 如果某个向量太短，角度计算不可靠。
        return 180.0  # 返回 180 度作为兜底值。

    cos_value = np.dot(ba, bc) / (norm_ba * norm_bc)  # 用点积公式计算夹角余弦值。
    cos_value = np.clip(cos_value, -1.0, 1.0)  # 限制范围，避免浮点误差导致 arccos 报错。
    return float(np.degrees(np.arccos(cos_value)))  # 把弧度转成角度并返回。


def point_in_roi(cx, cy, roi_cfg):  # 定义判断点是否在 ROI 区域内的函数。
    if not roi_cfg.get("enabled", False):  # 如果没有启用 ROI。
        return True  # 默认认为所有点都在有效区域内。

    return roi_cfg["x1"] <= cx <= roi_cfg["x2"] and roi_cfg["y1"] <= cy <= roi_cfg["y2"]  # 判断点是否落在 ROI 矩形内。


def play_audio_file(audio_path, max_seconds):  # 定义跨平台音频播放函数，支持 mp3。
    if not audio_path.exists():  # 如果音频文件不存在。
        print(f"[ERROR] Audio file not found: {audio_path}")  # 打印缺失文件路径。
        return

    if platform.system() == "Windows":  # Windows 使用 MediaPlayer，支持 mp3。
        script = (
            "Add-Type -AssemblyName PresentationCore; "
            "$player = New-Object System.Windows.Media.MediaPlayer; "
            "$player.Open([Uri]::new($args[0])); "
            "$deadline = (Get-Date).AddSeconds([double]$args[1]); "
            "while (-not $player.NaturalDuration.HasTimeSpan -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 100 }; "
            "$player.Play(); "
            "while ($player.NaturalDuration.HasTimeSpan -and $player.Position -lt $player.NaturalDuration.TimeSpan -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 100 }; "
            "$player.Stop(); "
            "$player.Close()"
        )
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", script, str(audio_path), str(max_seconds)],
            check=False,
            timeout=max_seconds + 1,
        )
        return

    if audio_path.suffix.lower() == ".mp3":  # Linux/树莓派上优先使用常见 mp3 播放器。
        commands = [
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(audio_path)],
            ["mpg123", "-q", str(audio_path)],
        ]
    else:
        commands = [
            ["aplay", str(audio_path)],
            ["paplay", str(audio_path)],
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(audio_path)],
        ]

    for command in commands:  # 选择系统中已安装的第一个播放器。
        if shutil.which(command[0]):
            subprocess.run(command, check=False, timeout=max_seconds)
            return

    print("[ERROR] No supported audio player found. Install ffmpeg/ffplay or mpg123 to play mp3 reminders.")


def speak(text, voice_cfg):  # 定义提醒函数，支持打印、音频文件和 espeak。
    mode = voice_cfg.get("mode", "print")  # 读取提醒模式，默认只打印。
    max_seconds = voice_cfg.get("max_seconds", 20)  # 读取单次提醒最长播放时间。
    if os.getenv("GUIDEBOT_EVENT_JSONL"):
        print(
            json.dumps(
                {
                    "label": "sedentary",
                    "sedentary": True,
                    "fatigue": False,
                    "message": text,
                    "confidence": 0.9,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if os.getenv("GUIDEBOT_DISABLE_LOCAL_REMINDER_AUDIO", "1") != "0":
            return
    print(f"[REMIND] {text}")  # 无论哪种模式，都先在终端打印提醒内容。

    if mode == "print":  # 如果是 print 模式。
        return  # 打印完就结束。

    if mode == "audio_file":  # 如果是播放音频文件模式。
        audio_file = voice_cfg.get("audio_file", "")  # 读取音频文件路径。
        if not audio_file:  # 如果没有配置音频文件。
            return  # 没有可播放内容，直接返回。
        audio_path = Path(audio_file)
        if not audio_path.is_absolute():
            audio_path = Path(__file__).resolve().parent / audio_path
        try:  # 尝试调用系统播放器。
            play_audio_file(audio_path, max_seconds)  # 播放配置的提醒音频。
        except subprocess.TimeoutExpired:  # 如果播放时间超过限制。
            print("[SYSTEM] Audio reminder timeout. Resume detection.")  # 打印自动恢复检测提示。
        except Exception as exc:  # 如果播放过程中出错。
            print(f"[ERROR] Failed to play audio file: {exc}")  # 打印错误原因。
        return  # audio_file 模式处理完成后返回。

    if mode == "system_tts":
        try:
            if platform.system() == "Windows":
                script = (
                    "Add-Type -AssemblyName System.Speech; "
                    "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                    "$speaker.Speak($args[0])"
                )
                subprocess.run(["powershell", "-NoProfile", "-Command", script, text], check=False, timeout=max_seconds)
            else:
                subprocess.run(["espeak-ng", text], check=False, timeout=max_seconds)
        except subprocess.TimeoutExpired:
            print("[SYSTEM] Voice reminder timeout. Resume detection.")
        except Exception as exc:
            print(f"[ERROR] Failed to run system TTS: {exc}")
        return

    if mode == "espeak":  # 如果是 espeak 语音合成模式。
        try:  # 尝试调用 espeak-ng。
            subprocess.run(["espeak-ng", "-v", "zh", text], check=False, timeout=max_seconds)  # 使用中文语音朗读提醒文本。
        except subprocess.TimeoutExpired:  # 如果语音播报时间超过限制。
            print("[SYSTEM] Voice reminder timeout. Resume detection.")  # 打印自动恢复检测提示。
        except Exception as exc:  # 如果 espeak-ng 调用失败。
            print(f"[ERROR] Failed to run espeak-ng: {exc}")  # 打印错误原因。


def select_target_person(boxes, keypoints, roi_cfg):  # 定义从多人中选择监测目标的函数。
    best_idx = None  # 保存当前最佳目标的索引，初始为空。
    best_area = 0  # 保存当前最大人体框面积，初始为 0。

    for i, box in enumerate(boxes):  # 遍历每一个检测到的人体框。
        x1, y1, x2, y2 = box  # 拆出人体框左上角和右下角坐标。
        w = x2 - x1  # 计算人体框宽度。
        h = y2 - y1  # 计算人体框高度。
        if w <= 0 or h <= 0:  # 如果框宽或框高不合法。
            continue  # 跳过这个异常检测框。

        cx = (x1 + x2) / 2  # 计算人体框中心点 x 坐标。
        cy = (y1 + y2) / 2  # 计算人体框中心点 y 坐标。
        if not point_in_roi(cx, cy, roi_cfg):  # 如果中心点不在 ROI 内。
            continue  # 跳过这个人，不作为监测目标。

        area = w * h  # 计算人体框面积。
        if area > best_area:  # 如果当前人体框比之前的最佳目标更大。
            best_area = area  # 更新最大面积。
            best_idx = i  # 更新最佳目标索引。

    if best_idx is None:  # 如果没有找到符合条件的人。
        return None, None, None  # 返回空目标。

    return best_idx, boxes[best_idx], keypoints[best_idx]  # 返回目标索引、目标框和目标关键点。


def keypoint_visible(kpts, kpt_conf, idx, min_conf):  # 定义判断某个关键点是否可靠的函数。
    if idx >= len(kpts):  # 如果关键点索引超出坐标数组范围。
        return False  # 认为该关键点不可用。
    if not valid_point(kpts[idx]):  # 如果关键点坐标无效。
        return False  # 认为该关键点不可用。
    if kpt_conf is None:  # 如果没有关键点置信度数组。
        return True  # 只根据坐标有效性判断可用。
    return idx < len(kpt_conf) and kpt_conf[idx] >= min_conf  # 同时检查索引范围和置信度阈值。


def is_sitting_from_pose(kpts, box, kpt_conf=None, min_conf=0.25):  # 定义根据人体姿态判断是否坐着的函数。
    x1, y1, x2, y2 = box  # 拆出人体框坐标。
    person_w = x2 - x1  # 计算人体框宽度。
    person_h = y2 - y1  # 计算人体框高度。
    if person_w <= 0 or person_h <= 0:  # 如果人体框尺寸异常。
        return False  # 无法可靠判断坐姿，返回非坐姿。

    bbox_hint = person_h / person_w < 2.2  # 坐姿时人体框通常没有站立时那么细长。

    left_hip = kpts[11]  # 取左髋关键点坐标。
    right_hip = kpts[12]  # 取右髋关键点坐标。
    left_knee = kpts[13]  # 取左膝关键点坐标。
    right_knee = kpts[14]  # 取右膝关键点坐标。
    left_ankle = kpts[15]  # 取左脚踝关键点坐标。
    right_ankle = kpts[16]  # 取右脚踝关键点坐标。

    knee_bent = False  # 记录膝盖是否明显弯曲，初始为 False。
    if (  # 开始判断左腿三个关键点是否都可靠。
        keypoint_visible(kpts, kpt_conf, 11, min_conf)  # 左髋是否可靠。
        and keypoint_visible(kpts, kpt_conf, 13, min_conf)  # 左膝是否可靠。
        and keypoint_visible(kpts, kpt_conf, 15, min_conf)  # 左脚踝是否可靠。
    ):  # 左腿关键点检查结束。
        left_angle = calc_angle(left_hip, left_knee, left_ankle)  # 计算左髋-左膝-左脚踝夹角。
        knee_bent = 55 <= left_angle <= 145  # 如果角度在该范围，认为左膝弯曲。

    if (  # 开始判断右腿三个关键点是否都可靠。
        keypoint_visible(kpts, kpt_conf, 12, min_conf)  # 右髋是否可靠。
        and keypoint_visible(kpts, kpt_conf, 14, min_conf)  # 右膝是否可靠。
        and keypoint_visible(kpts, kpt_conf, 16, min_conf)  # 右脚踝是否可靠。
    ):  # 右腿关键点检查结束。
        right_angle = calc_angle(right_hip, right_knee, right_ankle)  # 计算右髋-右膝-右脚踝夹角。
        knee_bent = knee_bent or 55 <= right_angle <= 145  # 任意一条腿膝盖弯曲即可作为坐姿证据。

    hip_points = []  # 保存可靠的髋部关键点。
    knee_points = []  # 保存可靠的膝盖关键点。
    for idx in (11, 12):  # 遍历左右髋关键点索引。
        if keypoint_visible(kpts, kpt_conf, idx, min_conf):  # 如果当前髋部关键点可靠。
            hip_points.append(kpts[idx])  # 加入髋部关键点列表。
    for idx in (13, 14):  # 遍历左右膝关键点索引。
        if keypoint_visible(kpts, kpt_conf, idx, min_conf):  # 如果当前膝盖关键点可靠。
            knee_points.append(kpts[idx])  # 加入膝盖关键点列表。

    hip_knee_close = False  # 记录髋部和膝盖在竖直方向是否接近。
    if hip_points and knee_points:  # 只有髋部和膝盖都至少有一个可靠点时才计算。
        hip_y = np.mean([p[1] for p in hip_points])  # 计算髋部关键点平均 y 坐标。
        knee_y = np.mean([p[1] for p in knee_points])  # 计算膝盖关键点平均 y 坐标。
        hip_knee_close = abs(knee_y - hip_y) < person_h * 0.45  # 坐姿时髋和膝在竖直方向通常比较接近。

    score = 0  # 初始化坐姿评分。
    if bbox_hint:  # 如果人体框比例像坐姿。
        score += 1  # 坐姿评分加 1。
    if knee_bent:  # 如果膝盖明显弯曲。
        score += 2  # 坐姿评分加 2，因为这是更强证据。
    if hip_knee_close:  # 如果髋部和膝盖较接近。
        score += 1  # 坐姿评分加 1。

    return score >= 2  # 分数达到 2 就认为是坐姿。


def draw_roi(frame, roi_cfg):  # 定义绘制 ROI 区域的函数。
    if not roi_cfg.get("enabled", False):  # 如果没有启用 ROI。
        return  # 不画 ROI，直接返回。

    x1 = int(roi_cfg["x1"])  # 读取 ROI 左上角 x 坐标并转为整数。
    y1 = int(roi_cfg["y1"])  # 读取 ROI 左上角 y 坐标并转为整数。
    x2 = int(roi_cfg["x2"])  # 读取 ROI 右下角 x 坐标并转为整数。
    y2 = int(roi_cfg["y2"])  # 读取 ROI 右下角 y 坐标并转为整数。
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)  # 在画面上画出 ROI 矩形。
    cv2.putText(  # 在画面上写 ROI 标签。
        frame,  # 要绘制文字的图像。
        "ROI",  # 显示的文字内容。
        (x1, max(20, y1 - 8)),  # 文字位置，避免 y 坐标太靠上。
        cv2.FONT_HERSHEY_SIMPLEX,  # OpenCV 内置字体。
        0.7,  # 字体大小。
        (255, 255, 0),  # 字体颜色。
        2,  # 字体线宽。
    )  # ROI 标签绘制结束。


def draw_status(frame, state_text, sitting_seconds, fps, monitor_enabled):  # 定义绘制状态文字的函数。
    mode_text = "monitor: ON" if monitor_enabled else "monitor: PAUSED"  # 根据监测开关生成状态文字。
    lines = [  # 准备要显示在画面左上角的多行文字。
        mode_text,  # 第一行显示监测开关。
        f"state: {state_text}",  # 第二行显示平滑后的坐姿状态。
        f"sitting: {int(sitting_seconds)}s",  # 第三行显示已经连续坐了多少秒。
        f"FPS: {int(fps)}",  # 第四行显示当前帧率。
        "F: pause/resume  Q: quit",  # 第五行显示按键说明。
    ]  # 状态文字列表结束。

    y = 30  # 第一行文字的 y 坐标。
    for line in lines:  # 遍历每一行要显示的状态文字。
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)  # 把状态文字画到画面上。
        y += 28  # 下一行文字往下移动 28 像素。


def draw_pose(frame, kpts, kpt_conf=None, min_conf=0.25):  # 定义绘制人体骨架的函数。
    for start, end in COCO_SKELETON:  # 遍历每一条骨架连接。
        if keypoint_visible(kpts, kpt_conf, start, min_conf) and keypoint_visible(kpts, kpt_conf, end, min_conf):  # 如果两端关键点都可靠。
            p1 = tuple(kpts[start].astype(int))  # 把起点关键点坐标转为整数元组。
            p2 = tuple(kpts[end].astype(int))  # 把终点关键点坐标转为整数元组。
            cv2.line(frame, p1, p2, (0, 255, 255), 2)  # 在两个关键点之间画黄色骨架线。

    for idx, point in enumerate(kpts):  # 遍历每一个关键点。
        if keypoint_visible(kpts, kpt_conf, idx, min_conf):  # 如果该关键点可靠。
            cv2.circle(frame, tuple(point.astype(int)), 3, (0, 0, 255), -1)  # 在关键点位置画红色实心圆。


def configure_camera(cap, camera_cfg):  # 定义配置摄像头参数的函数。
    fourcc = camera_cfg.get("fourcc", "MJPG").upper()  # 读取视频编码格式并转为大写。
    if fourcc and len(fourcc) == 4:  # 如果编码格式是 4 个字符。
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*fourcc))  # 设置摄像头视频编码格式。
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_cfg.get("frame_width", 640))  # 设置摄像头画面宽度。
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_cfg.get("frame_height", 480))  # 设置摄像头画面高度。


def main():  # 定义程序主函数。
    cfg = load_config()  # 读取 config.yaml；如果文件为空或不存在，就使用默认配置。
    if os.getenv("GUIDEBOT_EVENT_JSONL") or os.getenv("GUIDEBOT_HEADLESS"):
        cfg["camera"]["show_window"] = False
    camera_cfg = cfg["camera"]  # 取出摄像头相关配置。
    model_cfg = cfg["model"]  # 取出 YOLO Pose 模型相关配置。
    roi_cfg = cfg["roi"]  # 取出 ROI 监测区域配置。
    sitting_cfg = cfg["sitting"]  # 取出久坐检测相关配置。
    voice_cfg = cfg["voice"]  # 取出提醒相关配置。
    model_path = Path(model_cfg["model_path"])
    if not model_path.is_absolute():
        model_cfg["model_path"] = str(Path(__file__).resolve().parent / model_path)

    camera_id = camera_cfg.get("camera_id", 0)  # 获取摄像头编号。
    show_window = camera_cfg.get("show_window", True)  # 获取是否显示窗口的配置。
    sit_limit_seconds = sitting_cfg.get("sit_limit_seconds", 60)  # 获取久坐提醒阈值。
    reset_seconds = sitting_cfg.get("reset_seconds", 30)  # 获取离开坐姿后清零计时的阈值。
    remind_cooldown_seconds = sitting_cfg.get("remind_cooldown_seconds", 60)  # 获取提醒冷却时间。
    detect_interval = sitting_cfg.get("detect_interval", 0.5)  # 获取姿态检测时间间隔。
    smooth_window = sitting_cfg.get("smooth_window", 10)  # 获取平滑窗口大小。
    sitting_ratio_threshold = sitting_cfg.get("sitting_ratio_threshold", 0.6)  # 获取稳定坐姿比例阈值。
    keypoint_conf = model_cfg.get("keypoint_conf", 0.25)  # 获取关键点置信度阈值。

    print(f"[SYSTEM] Voice mode: {voice_cfg.get('mode', 'print')}.")  # 打印当前提醒模式，方便确认配置是否生效。
    print("[SYSTEM] Loading pose model...")  # 打印模型加载提示。
    backend, model = load_pose_model(model_cfg)  # 加载姿态模型。
    print(f"[SYSTEM] Pose model loaded with backend: {backend}.")  # 打印模型加载完成提示。

    cap = cv2.VideoCapture(camera_id)  # 打开摄像头。
    configure_camera(cap, camera_cfg)  # 设置摄像头参数。
    if not cap.isOpened():  # 如果摄像头没有成功打开。
        print("[ERROR] Failed to open camera. Check camera_id or camera permission.")  # 打印错误提示。
        return  # 结束主函数。

    sitting_history = deque(maxlen=smooth_window)  # 创建坐姿历史队列。
    seated_since = None  # 记录当前这段稳定坐姿开始时间。
    accumulated_sitting_seconds = 0.0  # 记录已经累计的坐姿时长，站起时会暂停在这里。
    non_seated_since = None  # 记录非坐姿开始时间。
    last_remind_time = 0  # 记录上一次提醒时间。
    last_detect_time = 0  # 记录上一次检测时间。
    last_frame_time = time.time()  # 记录上一帧时间，用于计算 FPS。

    monitor_enabled = True  # 设置监测初始状态为开启。
    stable_state = "unknown"  # 设置初始稳定状态为 unknown。
    raw_sitting = False  # 设置初始原始坐姿判断为 False。
    target_box = None  # 设置初始目标框为空。
    target_kpts = None  # 设置初始目标关键点为空。
    target_kpt_conf = None  # 设置初始目标关键点置信度为空。

    print("[SYSTEM] Health guardian started. Press F to pause/resume, Q to quit.")  # 打印启动提示。

    while cap.isOpened():  # 摄像头打开时持续处理视频帧。
        loop_start = time.time()  # 记录本轮循环开始时间。
        ret, frame = cap.read()  # 读取一帧摄像头画面。
        if not ret:  # 如果读取失败。
            print("[ERROR] Failed to read camera frame.")  # 打印读取失败提示。
            break  # 跳出主循环。

        now = time.time()  # 获取当前时间。
        fps = 1.0 / max(now - last_frame_time, 1e-6)  # 根据两帧间隔计算 FPS。
        last_frame_time = now  # 更新上一帧时间。

        if monitor_enabled and now - last_detect_time >= detect_interval:  # 如果监测开启且达到检测间隔。
            last_detect_time = now  # 更新最近检测时间。
            raw_sitting = False  # 重置本次原始坐姿判断。
            target_box = None  # 清空上一轮目标框。
            target_kpts = None  # 清空上一轮目标关键点。
            target_kpt_conf = None  # 清空上一轮关键点置信度。

            boxes, kpts_all, conf_all = predict_pose(model, backend, frame, model_cfg)  # 使用配置的后端做姿态预测。
            if len(boxes) > 0 and len(kpts_all) > 0:  # 如果至少检测到一个人。
                target_idx, target_box, target_kpts = select_target_person(boxes, kpts_all, roi_cfg)  # 选择监测目标。
                if target_idx is not None:  # 如果成功选出目标。
                    if conf_all is not None:  # 如果有关键点置信度。
                        target_kpt_conf = conf_all[target_idx]  # 取出目标人物的关键点置信度。
                    raw_sitting = is_sitting_from_pose(target_kpts, target_box, target_kpt_conf, keypoint_conf)  # 判断目标是否坐着。

            sitting_history.append(raw_sitting)  # 保存本次坐姿判断结果。
            enough_history = len(sitting_history) >= max(3, smooth_window // 2)  # 判断历史数量是否足够。
            sitting_ratio = sum(sitting_history) / len(sitting_history) if enough_history else 0  # 计算最近坐姿比例。
            stable_sitting = sitting_ratio >= sitting_ratio_threshold  # 判断是否达到稳定坐姿条件。

            if stable_sitting:  # 如果稳定状态是坐姿。
                stable_state = "sitting"  # 更新状态为 sitting。
                if seated_since is None:  # 如果刚开始稳定坐着。
                    seated_since = now  # 记录坐姿开始时间。
                    print("[STATE] Stable sitting detected. Start or resume sitting timer.")  # 打印开始或恢复计时提示。
                non_seated_since = None  # 清空非坐姿开始时间。

                sitting_duration = accumulated_sitting_seconds + (now - seated_since)  # 计算累计坐姿时长。
                if sitting_duration >= sit_limit_seconds and now - last_remind_time >= remind_cooldown_seconds:  # 判断是否需要提醒。
                    remind_texts = voice_cfg.get("remind_texts", [])  # 获取提醒文案列表。
                    text = random.choice(remind_texts) if remind_texts else "Please stand up and move."  # 选择提醒文案。
                    speak(text, voice_cfg)  # 执行提醒。
                    last_remind_time = time.time()  # 更新最近提醒时间，播放完后继续累计计时。
            else:  # 如果稳定状态不是坐姿。
                stable_state = "not_sitting"  # 更新状态为 not_sitting。
                if seated_since is not None:  # 如果刚从坐姿切换到非坐姿。
                    accumulated_sitting_seconds += now - seated_since  # 结算当前坐姿片段，让计时暂停。
                    seated_since = None  # 清空当前坐姿片段开始时间。
                    print("[STATE] User stood up. Pause sitting timer.")  # 打印暂停计时提示。
                if non_seated_since is None:  # 如果刚进入非坐姿状态。
                    non_seated_since = now  # 记录非坐姿开始时间。
                if accumulated_sitting_seconds > 0 and now - non_seated_since >= reset_seconds:  # 如果已经离开坐姿足够久。
                    print("[STATE] User left or stood up long enough. Reset sitting timer.")  # 打印重置提示。
                    accumulated_sitting_seconds = 0.0  # 清空累计坐姿时间。
                    last_remind_time = 0  # 清空提醒时间。

        sitting_seconds = accumulated_sitting_seconds  # 先取已经累计的坐姿秒数。
        if seated_since is not None:  # 如果当前正在坐着。
            sitting_seconds += time.time() - seated_since  # 加上当前坐姿片段的时长。

        draw_roi(frame, roi_cfg)  # 绘制 ROI 区域。
        if target_box is not None:  # 如果当前有目标人物框。
            x1, y1, x2, y2 = target_box.astype(int)  # 把目标框坐标转成整数。
            color = (0, 255, 0) if raw_sitting else (0, 0, 255)  # 根据坐姿状态选择框颜色。
            label = "raw sitting" if raw_sitting else "raw not sitting"  # 根据坐姿状态生成标签。
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # 绘制目标人物框。
            cv2.putText(frame, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)  # 绘制目标状态标签。
        if target_kpts is not None:  # 如果当前有目标人物关键点。
            draw_pose(frame, target_kpts, target_kpt_conf, keypoint_conf)  # 绘制人体骨架。

        draw_status(frame, stable_state, sitting_seconds, fps, monitor_enabled)  # 绘制监测状态信息。

        if show_window:  # 如果允许显示窗口。
            cv2.imshow("AI Car Health Guardian", frame)  # 显示处理后的画面。
            action = cv2.waitKey(1) & 0xFF  # 读取键盘输入。
            if action in (ord("q"), ord("Q")):  # 如果按下 q 或 Q。
                break  # 退出主循环。
            if action in (ord("f"), ord("F")):  # 如果按下 f 或 F。
                monitor_enabled = not monitor_enabled  # 切换监测开关。
                print(f"[SYSTEM] Monitor {'resumed' if monitor_enabled else 'paused'}.")  # 打印切换后的状态。

        elapsed = time.time() - loop_start  # 计算本轮循环耗时。
        if elapsed < 0.001:  # 如果循环过快。
            time.sleep(0.001)  # 短暂休眠，避免 CPU 空转。

    cap.release()  # 释放摄像头资源。
    cv2.destroyAllWindows()  # 关闭 OpenCV 创建的所有窗口。
    print("[SYSTEM] Program finished.")  # 打印程序结束提示。


if __name__ == "__main__":  # 判断当前文件是否被直接运行。
    main()  # 直接运行本文件时启动主函数。
