# 原厂小车程序兼容性与更新建议

## 审计范围

- `project_demo/09.AI_Big_Model`
- `project_demo/10.Basic_voice_control`
- `project_demo/03.Basic_car_course` 至 `08.AI_Visual_Interaction_Course`
- `project_demo/lib/McLumk_Wheel_Sports.py`
- `project_demo/raspbot/Raspbot_Lib.py`

本次只读原厂目录。所有新实现继续放在 Guidebot 仓库内。

## 语音兼容性结论

| 原厂能力 | Guidebot 当前能力 | 兼容状态 |
|---|---|---|
| `/dev/myspeech`, 115200 baud | `YahboomSerialCommandSource` | 兼容；按 `AA 55 lang command FB` 同步 |
| 命令 ID 1–9 | 默认中文文本映射 | 兼容；其余 ID 需从协议表注入 |
| 09 的 16 kHz/mono/PCM16 麦克风 | `VoiceConfig` + `ArecordAudioSource` | 音频格式兼容 |
| PyAudio 固定 FFT 阈值录音 | `TurnCapture` + VAD Protocol | 可替换，不直接复用实现 |
| 讯飞/通义 WAV STT | `SpeechToText` Protocol | 接口兼容，需要单独 backend；当前未接云 API |
| 讯飞 MP3/通义 WAV TTS | `TextToSpeech` Protocol | 接口兼容，需要解码成 PCM 或专用播放器 backend |
| `mplayer` + `pkill mplayer` 打断 | 流式 player + task cancel | 设计兼容，新实现更安全 |
| 关键词唤醒后自由录音 | 串口事件 + PCM pipeline | 组件齐备，唤醒编排 loop 尚待实现 |
| 语音命令直接驱动电机 | 文本进入 Guidebot Agent OS | 尚不兼容：缺运动 Skill 与真实 DeviceAdapter |

因此当前语音层可以替换原厂的采集/识别编排，但还不能直接让真车运动。必须先实现
`RaspbotDeviceAdapter`，并让文本命令经过 Router、SafetyPolicy 和结构化 `Action`。

## 09.AI_Big_Model 迁移判断

值得保留的是能力边界，而不是原实现：

- 16 kHz 单声道录音、唤醒后开始一轮自由对话。
- 决策层与执行层分离的思路。
- 颜色跟随、巡线、人脸跟随、避障、灯效和音乐等动作集合。
- 唤醒时停止运动、视觉子进程和播放的紧急复位语义。

不应直接迁入 Guidebot 的部分：

- 使用 `eval()` 执行模型返回字符串，包括所谓 `safe_eval()`；它并不安全。
- API key 通过 Python 模块导入、关闭 TLS 证书校验。
- 大量全局变量、daemon thread、固定相对路径和 `time.sleep()`。
- 用 `pgrep/pkill` 按名称杀进程，可能误杀其他会话。
- 先播报“开始执行”，再发现动作非法；确认语应在 SafetyPolicy 接受后产生。
- 每次动作前固定拍照并阻塞，摄像头没有统一所有权。

迁移时应把模型输出限制为 JSON Schema tool call，由工具注册表映射为 `Action`，禁止任意代码
求值。动作必须经过 Guidebot 的不可进化安全门。

## 10.Basic_voice_control 迁移判断

命令 ID 可作为测试和兼容输入：

- 1–3：停止/休眠类命令。
- 4–9：前进、后退、平移、旋转。
- 10–17：RGB 关闭、颜色与灯效。
- 22–26：停止巡线及红/绿/蓝/黄线巡航。
- 60：颜色识别播报。
- 71–76：人脸/颜色跟随与停止。

这些脚本直接开线程、控制 I2C 或启动相对路径子进程，不适合作为 Agent Skill。应将 ID 转换为
结构化 intent，再由 Skill 产生动作；长期运行的巡线/跟随任务应由统一的 `BehaviorSupervisor`
负责启动、取消、超时和复位。

## 可更新控制模块

### P0：硬件抽象与运动安全

1. 新建 `RaspbotDeviceAdapter`，封装原厂 I2C 地址 `0x2B`，不要修改 `Raspbot_Lib.py`。
2. 将 Guidebot `speed ∈ [0,1]` 映射为电机 `0..255`，方向使用 Enum，持续时间设置硬上限。
3. 所有运动使用 `try/finally` 自动停车，并增加独立 watchdog/E-stop。
4. I2C 访问加单进程锁；原代码会在多个模块各自创建 `Raspbot()`，可能竞争同一总线。
5. 新 driver 抛出类型化错误，不沿用原库吞掉全部异常的行为。

原 `Raspbot_Lib` 还存在“先构造 data、后裁剪参数”的问题，例如 Servo/RGB 的裁剪不会进入
已构造的数据数组；新 adapter 必须在调用前完成校验。

### P1：传感器事件源

- 超声波寄存器 `0x1A/0x1B` → 距离 Reading，并成为 MOVE 的动态安全条件。
- 四路巡线寄存器 `0x0A` → line sensor Reading。
- 红外寄存器 `0x0C`、按键 → 用户输入 Event。
- 轮询放入单一 async service，避免各玩法各开一个永久线程。

### P2：灯光、蜂鸣器和舵机

- RGB/灯效 → DISPLAY action；效果必须可取消，不能使用无界 while loop。
- 蜂鸣器 → 独立 feedback action，不冒充语音 SPEAK。
- 舵机 → POSE action，明确水平/俯仰机械限位和中位复位。
- 设备资源使用 ownership/lock，避免视觉跟踪和“点头”同时控制云台。

### P3：视觉行为监督器

- 摄像头由单一 capture service 管理，视觉节点消费帧，禁止多个 `VideoCapture(0)` 并发。
- 颜色/人脸/巡线算法发布结构化 observation，不直接写四个电机。
- PID 控制器作为低层 behavior，启动/停止由 supervisor 管理。
- 替换 `python ./script.py`、`pgrep`、`kill` 为带 PID handle、超时和 cleanup 的子进程或任务。

### P4：大模型工具层

- 将原决策提示词中的动作变成有 schema 的工具：move、stop、set_light、set_pose、track、patrol。
- 参数解析、单位换算和边界检查在确定性代码中完成。
- 模型只能提出 Action，不能访问 I2C、shell、文件系统或 SafetyPolicy。

## 推荐落地顺序

```text
Yahboom command/PCM voice
  → Guidebot text event
  → intent Skill
  → SafetyPolicy
  → RaspbotDeviceAdapter
  → motor/light/servo
  → telemetry feedback
```

先实现 stop/forward/back/left/right 五个动作和超声波安全停车，再接灯光与舵机，最后接巡线和
视觉跟随。这样每一步都能用模拟 driver 测试，不需要在开发阶段冒险让真实小车执行未验证动作。
