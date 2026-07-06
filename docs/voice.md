# Guidebot 语音模块

## 目标与替换边界

语音模块位于 `src/guidebot/voice/`，是全新实现，不修改或复制
`/workspace/ylj/harry_main/bot` 中的原厂课程代码。删除该目录即可完整移除语音能力；替换任一
后端不需要修改 `GuidebotHub`。

```text
PCM AudioSource → VAD → TurnCapture → STT → GuidebotDialogueAdapter
                                              ↓
Speaker ← streaming TTS ← response text ← GuidebotHub.say
```

接口分别定义在 `interfaces.py`：

- `AudioSource`：读取固定格式 PCM frame。
- `VoiceActivityDetector`：判断单帧是否为语音。
- `SpeechToText`：完整轮次转写。
- `DialogueBackend`：文本请求/响应；Guidebot adapter 复用现有 Agent OS。
- `TextToSpeech`：异步产出音频块。
- `AudioPlayer`：播放和立即停止。

默认实现只用于无硬件演示和测试，不引入模型依赖。生产环境可逐个替换成 Silero VAD、Whisper
或云端流式 STT、Kokoro/云端 TTS 和 ALSA 播放器。

树莓派标准 ALSA 路径已提供 `ArecordAudioSource` 与 `AplayAudioPlayer`，二者通过持久子进程读写
raw PCM，不要求 `sounddevice`。可用 `device="plughw:1,0"` 显式选择声卡；实际编号以
`arecord -l` 和 `aplay -l` 为准。

## 原厂语音模块适配

随车 `Speech_Lib` v0.0.3 使用 `/dev/myspeech`、115200 baud。它不是自由语音转写设备，而是
离线关键词识别器：完整帧为 `AA 55 <语言/唤醒> <命令 ID> FB`，原库随后调用 `mplayer`
播放预录 MP3。新解析器会校验帧头/帧尾、累积 USB 串口短读，并在噪声字节后重新同步。

新实现 `YahboomSerialCommandSource` 只解析串口协议，不调用原库、不使用 shell 播放器。默认
映射了停止、休眠、前后移动和转向 ID，完整映射可以由构造参数注入：

```python
from guidebot.voice.yahboom_serial import YahboomSerialCommandSource

source = YahboomSerialCommandSource(
    "/dev/myspeech",
    command_text={1: "停止", 4: "前进", 5: "后退"},
)
text = await source.read_text()
```

安装唯一的硬件可选依赖：

```bash
pip install -e '.[voice-hardware]'
```

## PCM 流水线配置

推荐格式为 16 kHz、单声道、16-bit little-endian PCM、20 ms/frame。`VoiceConfig` 默认设置：

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `pre_roll_ms` | 300 | 保留触发前音频，防止切掉首词 |
| `min_speech_ms` | 250 | 过滤短噪声 |
| `silence_hangover_ms` | 500 | 轮次结束判定 |
| `max_turn_ms` | 20000 | 防止无限录音 |

`VoicePipeline.interrupt()` 会取消当前流式 TTS 并调用播放器的 `stop()`。部署时应让独立 VAD
监听器在扬声器播放期间持续工作，检测到用户插话后调用该方法。麦克风与扬声器共处一机时还需
使用系统 AEC（回声消除），否则机器人自己的 TTS 可能触发打断。

当 dialogue backend 提供 `stream(text)` 时，`VoicePipeline` 会并行执行 LLM token 生产和 TTS
消费。`TextSegmenter` 将 token 合并为自然短句后立即提交 TTS；它不是等完整回答生成完再播放。
`BargeInMonitor` 在播放期间使用独立 VAD 检测插话并取消整个生成/播放任务。

## 快速验证

```bash
guidebot voice-demo
pytest tests/test_voice_*.py
```

`voice-demo` 使用脚本音频、能量 VAD、脚本 STT 和内存播放器，只验证编排与接口，不声称代表
真实识别质量。

## 部署顺序

1. 在小车上确认 `arecord -l`、`aplay -l` 和 `/dev/myspeech` 是否存在。
2. 先跑原厂关键词串口 adapter，验证协议和设备权限。
3. 接入 USB/I2S 麦克风 AudioSource，再替换 Silero VAD 与 STT。
4. 接入 TTS 和 ALSA Player，测量首音频延迟。
5. 最后启用独立打断监听与 AEC。

涉及移动、灯光等命令时，文本仍需进入 Guidebot Agent OS，由 Router 和不可进化的
`SafetyPolicy` 决定是否执行；语音模块不能直接控制底盘。

生产后端选型、情感对话设计和树莓派/GPU 部署拓扑见
[realtime-voice-deployment.md](realtime-voice-deployment.md)。
