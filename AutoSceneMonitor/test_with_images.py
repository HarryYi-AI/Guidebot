import argparse
import json
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from auto_scene_monitor import (  # noqa: E402
    DEFAULT_VISION_MODEL,
    DEFAULT_WATCH_RULE,
    DEFAULT_WATCH_RULE_FILE,
    describe_scene,
    judge_scene,
    load_text_file,
    speak_feedback,
    write_log,
)


DEFAULT_TEST_IMAGE_DIR = BASE_DIR / "test_images"
DEFAULT_RESULT_FILE = BASE_DIR / "logs" / "image_test_results.jsonl"


def iter_images(image_dir):
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(
        path for path in Path(image_dir).iterdir() if path.suffix.lower() in suffixes
    )


def load_sources(image_dir):
    source_file = Path(image_dir) / "sources.json"
    if not source_file.exists():
        return {}

    items = json.loads(source_file.read_text(encoding="utf-8"))
    return {Path(item["file"]).name: item for item in items}


def run_image_test(args):
    image_dir = Path(args.images_dir)
    images = iter_images(image_dir)
    if not images:
        raise RuntimeError(f"没有找到测试图片：{image_dir}")

    watch_rule = load_text_file(args.watch_rule_file, DEFAULT_WATCH_RULE)
    sources = load_sources(image_dir)
    args.result_file.parent.mkdir(parents=True, exist_ok=True)

    for image_path in images:
        print("\n" + "=" * 60)
        print(f"测试图片：{image_path}")

        scene_description = describe_scene(
            image_path=image_path,
            prompt=args.scene_prompt,
            model=args.vision_model,
        )
        judge_result = judge_scene(
            scene_description=scene_description,
            watch_rule=watch_rule,
            model=args.judge_model,
        )

        record = {
            "image_path": str(image_path),
            "source": sources.get(image_path.name, {}),
            "scene_description": scene_description,
            "judge_result": judge_result,
        }
        write_log(record)
        with open(args.result_file, "a", encoding="utf-8") as result_file:
            result_file.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"场景描述：{scene_description}")
        print(f"异常判断：{json.dumps(judge_result, ensure_ascii=False)}")

        if args.speak and judge_result.get("abnormal"):
            feedback = judge_result.get("feedback") or "检测到异常，请及时查看当前环境。"
            print(f"语音播报：{feedback}")
            speak_feedback(feedback)


def parse_args():
    parser = argparse.ArgumentParser(description="用下载图片模拟摄像头照片，测试异常规则。")
    parser.add_argument("--images-dir", default=str(DEFAULT_TEST_IMAGE_DIR))
    parser.add_argument("--vision-model", default=DEFAULT_VISION_MODEL)
    parser.add_argument("--judge-model", default="")
    parser.add_argument("--watch-rule-file", default=str(DEFAULT_WATCH_RULE_FILE))
    parser.add_argument(
        "--scene-prompt",
        default="请客观描述图片中的室内/周边环境，重点说明是否有火焰、烟雾、积水、人员倒地、门窗未关、窗户破损、通道堵塞、遮挡或设备异常。",
    )
    parser.add_argument("--result-file", type=Path, default=DEFAULT_RESULT_FILE)
    parser.add_argument(
        "--speak",
        dest="speak",
        action="store_true",
        default=True,
        help="测试图片判断为异常时进行语音播报，默认开启。",
    )
    parser.add_argument(
        "--no-speak",
        dest="speak",
        action="store_false",
        help="关闭测试图片异常语音播报，仅打印文字结果。",
    )
    args = parser.parse_args()
    if not args.judge_model:
        args.judge_model = args.vision_model
    return args


if __name__ == "__main__":
    run_image_test(parse_args())
