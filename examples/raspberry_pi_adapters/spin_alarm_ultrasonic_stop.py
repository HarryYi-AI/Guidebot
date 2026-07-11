#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
原地小范围旋转吵闹闹钟 v2：超声波挡停版（Raspbot / 麦克纳姆轮小车）

目标：
- 不再要求按车体 KEY1，避免为了关闹钟去接触小车。
- 闹钟触发后：蜂鸣器吵闹 + RGB 闪灯 + 小角度左右原地旋转。
- 停止方式：用手/人挡住前方超声波传感器，持续一小段时间后自动停止。

源码依据：
- McLumk_Wheel_Sports.py: rotate_left / rotate_right / stop_robot / bot
- 1.Buzzer driver.ipynb: bot.Ctrl_BEEP_Switch(1/0)
- 3.RGB Light bar test.ipynb: bot.Ctrl_WQ2812_ALL(1, color) / bot.Ctrl_WQ2812_ALL(0, 0)
- 6.Ultrasonic distance measurement.ipynb:
    bot.Ctrl_Ulatist_Switch(1/0)
    diss_H = bot.read_data_array(0x1b, 1)[0]
    diss_L = bot.read_data_array(0x1a, 1)[0]
    dis = diss_H << 8 | diss_L

运行示例：
  # 10 秒后开始测试，速度更低、旋转更小
  python3 spin_alarm_ultrasonic_stop_v2.py --demo 10 --speed 25 --run-duration 60

  # 立即开始
  python3 spin_alarm_ultrasonic_stop_v2.py --now --speed 25

  # 早上 07:30 触发
  python3 spin_alarm_ultrasonic_stop_v2.py --alarm 07:30 --speed 25

停止方式：
- 用手靠近/挡住超声波前方，默认距离 <= 180mm 且持续 0.70 秒；或
- 终端 Ctrl-C。
"""

import argparse
import datetime as _dt
import os
import random
import sys
import time
from typing import Optional

# 兼容官方 notebook 的路径写法。
sys.path.insert(0, os.getcwd())
sys.path.append('/home/pi/project_demo/lib')
sys.path.append('/home/pi/project_demo/09.AI_Big_Model/AI_CarAgent')
sys.path.append('/home/pi/project_demo/04.Car_motion_control')

try:
    import McLumk_Wheel_Sports as car
except Exception as exc:  # noqa: BLE001
    print("无法导入 McLumk_Wheel_Sports。请把本脚本放到含有 McLumk_Wheel_Sports.py 的目录，")
    print("或确认 /home/pi/project_demo/lib、04.Car_motion_control、09.AI_Big_Model/AI_CarAgent 中已有该文件。")
    raise exc


def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(x)))


def set_beep(on: bool) -> None:
    try:
        car.bot.Ctrl_BEEP_Switch(1 if on else 0)
    except Exception:
        pass


def set_rgb_color(color: int) -> None:
    """color 按官方示例：0红 1绿 2蓝 3黄 4紫 5青 6白。"""
    try:
        car.bot.Ctrl_WQ2812_ALL(1, int(color) % 7)
    except Exception:
        pass


def rgb_off() -> None:
    try:
        car.bot.Ctrl_WQ2812_ALL(0, 0)
    except Exception:
        pass


def ultrasonic_on() -> None:
    try:
        car.bot.Ctrl_Ulatist_Switch(1)
        time.sleep(0.10)  # 官方示例里也给传感器一点测量时间
    except Exception:
        pass


def ultrasonic_off() -> None:
    try:
        car.bot.Ctrl_Ulatist_Switch(0)
    except Exception:
        pass


def read_ultrasonic_mm() -> Optional[int]:
    """读取前方超声波距离，单位 mm；失败或明显异常时返回 None。"""
    try:
        diss_h = car.bot.read_data_array(0x1B, 1)[0]
        diss_l = car.bot.read_data_array(0x1A, 1)[0]
        dis = (int(diss_h) << 8) | int(diss_l)
    except Exception:
        return None

    # 超声波偶尔可能返回 0 或离谱值，这里保守过滤，避免误关闹钟。
    if dis <= 0 or dis > 5000:
        return None
    return dis


def all_stop() -> None:
    """确保电机、蜂鸣器、灯、超声波都关闭。"""
    try:
        car.stop_robot()
    except Exception:
        pass
    set_beep(False)
    rgb_off()
    ultrasonic_off()


class UltrasonicStopDetector:
    """用“距离小于阈值且持续一段时间”作为停止手势。"""

    def __init__(self, stop_mm: int, hold_secs: float, reset_gap_secs: float = 0.25) -> None:
        self.stop_mm = int(stop_mm)
        self.hold_secs = float(hold_secs)
        self.reset_gap_secs = float(reset_gap_secs)
        self.first_block_time: Optional[float] = None
        self.last_block_time: Optional[float] = None

    def update(self, now: float) -> tuple[bool, Optional[int], float]:
        dis = read_ultrasonic_mm()
        blocked = dis is not None and dis <= self.stop_mm

        if blocked:
            if self.first_block_time is None:
                self.first_block_time = now
            self.last_block_time = now
        else:
            # 如果短暂丢一次读数，不立刻清零；超过 reset_gap 才重置。
            if self.last_block_time is None or now - self.last_block_time > self.reset_gap_secs:
                self.first_block_time = None
                self.last_block_time = None

        held = 0.0 if self.first_block_time is None else now - self.first_block_time
        return held >= self.hold_secs, dis, held


def wait_until_start(demo: Optional[int], alarm: Optional[str], now: bool) -> None:
    if now or (demo is None and alarm is None):
        return

    if demo is not None:
        seconds = max(0, int(demo))
        print(f"demo 模式：{seconds} 秒后开始。Ctrl-C 可取消。")
        end = time.time() + seconds
        while time.time() < end:
            time.sleep(0.1)
        return

    hour, minute = [int(x) for x in alarm.split(":")]
    now_dt = _dt.datetime.now()
    target = now_dt.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now_dt:
        target += _dt.timedelta(days=1)
    print(f"闹钟将在 {target.strftime('%Y-%m-%d %H:%M:%S')} 触发。Ctrl-C 可取消。")

    while _dt.datetime.now() < target:
        time.sleep(0.5)


def noisy_sleep(
    duration: float,
    beep_mode: str,
    rgb: bool,
    stop_detector: UltrasonicStopDetector,
    print_distance: bool,
) -> bool:
    """
    分片等待：蜂鸣/RGB/超声波挡停检测。
    返回 True 表示检测到挡停手势。
    """
    t_end = time.time() + max(0.0, duration)
    tick = 0
    last_print = 0.0

    while time.time() < t_end:
        now = time.time()
        should_stop, dis, held = stop_detector.update(now)
        if print_distance and now - last_print > 0.30:
            if dis is None:
                print("ultra: None")
            else:
                print(f"ultra: {dis:4d} mm, block_hold={held:.2f}s")
            last_print = now
        if should_stop:
            return True

        if beep_mode == "continuous":
            set_beep(True)
        else:
            # 脉冲蜂鸣：避免蜂鸣器长时间单调连续工作。
            set_beep(tick % 3 != 2)

        if rgb:
            set_rgb_color(tick)

        time.sleep(0.05)
        tick += 1
    return False


def spin_alarm_loop(
    speed: int,
    run_duration: int,
    min_spin: float,
    max_spin: float,
    pause: float,
    beep_mode: str,
    rgb: bool,
    stop_mm: int,
    hold_secs: float,
    print_distance: bool,
) -> None:
    """
    小范围原地旋转版：只调用 rotate_left/rotate_right，不调用 move_forward/move_left 等平移函数。
    停止方式改为超声波挡停，不再要求按车体按键。
    """
    # 进一步缩小范围：默认速度和单次旋转时间都比 v1 更低。
    speed = clamp_int(speed, 0, 80)
    min_spin = max(0.03, float(min_spin))
    max_spin = max(min_spin, float(max_spin))
    pause = max(0.0, float(pause))
    run_duration = max(1, int(run_duration))
    stop_mm = clamp_int(stop_mm, 30, 1000)
    hold_secs = max(0.10, float(hold_secs))

    stop_detector = UltrasonicStopDetector(stop_mm=stop_mm, hold_secs=hold_secs)

    print("闹钟触发：小范围原地旋转 + 蜂鸣。")
    print("停止方式：用手/人挡住前方超声波传感器，或 Ctrl-C。")
    print(
        f"参数：speed={speed}, run_duration={run_duration}s, "
        f"spin={min_spin:.2f}~{max_spin:.2f}s, pause={pause:.2f}s, "
        f"stop_mm<={stop_mm}, hold_secs={hold_secs:.2f}"
    )

    ultrasonic_on()
    start = time.time()
    try:
        while time.time() - start < run_duration:
            # 先停住检测一下，避免用户已经把手放到前方时车还继续动。
            if noisy_sleep(0.05, beep_mode=beep_mode, rgb=rgb, stop_detector=stop_detector, print_distance=print_distance):
                print("检测到超声波挡停手势，停止闹钟。")
                break

            direction_left = random.random() < 0.5
            spin_time = random.uniform(min_spin, max_spin)

            if direction_left:
                car.rotate_left(speed)
            else:
                car.rotate_right(speed)

            should_stop = noisy_sleep(
                spin_time,
                beep_mode=beep_mode,
                rgb=rgb,
                stop_detector=stop_detector,
                print_distance=print_distance,
            )
            car.stop_robot()

            if should_stop:
                print("检测到超声波挡停手势，停止闹钟。")
                break

            if pause > 0:
                should_stop = noisy_sleep(
                    pause,
                    beep_mode=beep_mode,
                    rgb=rgb,
                    stop_detector=stop_detector,
                    print_distance=print_distance,
                )
                if should_stop:
                    print("检测到超声波挡停手势，停止闹钟。")
                    break

        else:
            print("达到最大运行时长，自动停止。")
    finally:
        all_stop()
        print("已关闭电机、蜂鸣器、RGB 灯和超声波。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="原地小范围旋转吵闹闹钟 v2：超声波挡停版")
    start_group = parser.add_mutually_exclusive_group()
    start_group.add_argument("--now", action="store_true", help="立即开始；默认也是立即开始")
    start_group.add_argument("--demo", type=int, default=None, help="N 秒后开始，测试用")
    start_group.add_argument("--alarm", type=str, default=None, help="闹钟时间，格式 HH:MM，例如 07:30")

    parser.add_argument("--speed", type=int, default=25, help="旋转速度，缩小范围建议 18~30")
    parser.add_argument("--run-duration", type=int, default=180, help="最多运行秒数")
    parser.add_argument("--min-spin", type=float, default=0.08, help="单次旋转最短秒数")
    parser.add_argument("--max-spin", type=float, default=0.20, help="单次旋转最长秒数")
    parser.add_argument("--pause", type=float, default=0.12, help="每段旋转后的短暂停顿秒数")
    parser.add_argument("--stop-mm", type=int, default=180, help="超声波距离 <= 该值视为有人/手挡住前方，单位 mm")
    parser.add_argument("--hold-secs", type=float, default=0.70, help="持续挡住这么久才停止，避免误触发")
    parser.add_argument("--beep-mode", choices=["pulse", "continuous"], default="pulse", help="蜂鸣模式")
    parser.add_argument("--no-rgb", action="store_true", help="关闭 RGB 闪灯")
    parser.add_argument("--print-distance", action="store_true", help="打印超声波距离，调试阈值时打开")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        wait_until_start(demo=args.demo, alarm=args.alarm, now=args.now)
        spin_alarm_loop(
            speed=args.speed,
            run_duration=args.run_duration,
            min_spin=args.min_spin,
            max_spin=args.max_spin,
            pause=args.pause,
            beep_mode=args.beep_mode,
            rgb=not args.no_rgb,
            stop_mm=args.stop_mm,
            hold_secs=args.hold_secs,
            print_distance=args.print_distance,
        )
    except KeyboardInterrupt:
        all_stop()
        print("已手动停止。")


if __name__ == "__main__":
    main()
