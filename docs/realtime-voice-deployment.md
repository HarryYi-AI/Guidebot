# 实时与情感语音部署方案

## 当前实现状态

Guidebot 已实现级联语音链路的核心编排：

```text
PCM → VAD/turn detection → STT → streaming LLM tokens
    → semantic segment queue → streaming TTS → player
                              ↘ configurable barge-in policy
```

LLM token producer 与 TTS consumer 并发运行。语义分段器在句末标点或长度上限处提交短句，避免
逐 token 合成导致韵律破碎。`first_audio_latency_ms` 会记录从开始生成回答到首音频块的延迟。

尚未绑定具体 STT/LLM/TTS 服务，这是有意的：树莓派部署可按网络、预算和隐私要求选择后端，
Guidebot 的工具动作和 SafetyPolicy 不随语音供应商变化。

## 已接入：Qwen3.5 Omni Realtime

`DashScopeRealtimeSession` 已将阿里云回调式 SDK 封装为 Guidebot 的异步原生语音接口。数据路径为：

```text
arecord (16 kHz/mono/PCM16)
 → DashScope WebSocket + semantic VAD
 → Qwen3.5 Omni Realtime
 → 24 kHz/mono/PCM16 chunks → persistent aplay
```

它会输出用户与 Guidebot 的最终转录。插话默认使用两阶段 `transcript` 策略：先在
`speech_started` 时立即停止本地喇叭，避免 Guidebot 盖住用户说话；随后只有出现足够长的转写文本
或明确的“停一下/等一下”等短指令时，才取消正在生成的回复。这样比只依赖 `speech_started`
取消模型更不容易被咳嗽、清嗓子或喇叭回声误触发。API Key 只读取
`DASHSCOPE_API_KEY`；代码和命令行参数都不接收明文密钥。

### 树莓派 5 安装

```bash
cd /home/pi/Guidebot
python3 -m venv .venv
. .venv/bin/activate
sudo apt update
sudo apt install -y alsa-utils
pip install -e '.[voice-qwen]'

arecord -l
aplay -l
export DASHSCOPE_API_KEY='替换为已轮换的新密钥'
guidebot voice-qwen --voice Tina
```

若默认声卡不对，先用 `arecord -D plughw:1,0 -f S16_LE -r 16000 -c 1 /tmp/test.raw`
和 `aplay -D plughw:1,0 -f S16_LE -r 16000 -c 1 /tmp/test.raw` 做回环，再运行：

```bash
guidebot voice-qwen \
  --input-device plughw:1,0 \
  --output-device plughw:1,0 \
  --voice Tina
```

在 RASPBOT V2 的开放麦场景中，建议启用本地输入门控、唤醒词和 transcript 插话策略：

```bash
guidebot voice-qwen \
  --voice Tina \
  --input-device plughw:2,0 \
  --output-device default \
  --connect-retries 5 \
  --require-wake \
  --wake-phrase 你好小云 \
  --sleep-phrase 今天聊到这里 \
  --vad-threshold 0.75 \
  --vad-silence-ms 900 \
  --input-gate-rms 700 \
  --input-gate-hangover-ms 250 \
  --barge-in-mode transcript \
  --barge-in-min-chars 6 \
  --barge-in-early-stop-ms 1200
```

如果环境很吵、咳嗽仍会误触发，可以先完全关闭播报期间插话：

```bash
guidebot voice-qwen ... --barge-in-mode off
```

如果你希望最快速插话，才使用 `--barge-in-mode vad`；它延迟最低，但最容易被非语言声音误触发。

不要把 `export` 写进 Git 管理的文件。生产环境使用权限为 `0600` 的 systemd
`EnvironmentFile` 或云端 secret manager，并限制服务账户权限。当前 provider 默认使用低延迟模式，
不启用联网搜索；需要搜索时显式加 `--search`。若下一阶段接入小车 Function Calling，仍要让
tool call 经过 Guidebot `SafetyPolicy → Device`，绝不能让模型直接写串口或发布底盘速度。

若终端出现 `å®¸è¶...` 这类中文乱码，通常是树莓派 shell/SSH 客户端没有使用 UTF-8，可先运行：

```bash
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONUTF8=1
guidebot voice-qwen --voice Tina
```

开机自启可使用仓库中的 `deploy/guidebot-voice.service`：

```bash
sudo install -d -m 700 /etc/guidebot
printf 'DASHSCOPE_API_KEY=%s\n' "$DASHSCOPE_API_KEY" | sudo tee /etc/guidebot/voice.env >/dev/null
sudo chmod 600 /etc/guidebot/voice.env
sudo install -m 644 deploy/guidebot-voice.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now guidebot-voice
journalctl -u guidebot-voice -f
```

这条托管 API 路线不需要自建 GPU 服务器：树莓派通过 TLS WebSocket 直接连接北京地域服务。
只有选择自托管 Qwen/Moshi 时，才需要另配 GPU 服务器。

### 为什么这已经是“边听边想边说”

这不是旧程序的“整段录音 → ASR → 整段回答 → TTS”。麦克风 PCM 持续上传，服务端语义 VAD
判断轮次，模型以 `response.audio.delta` 持续返回语音，首个音频块抵达即可播放；文本转录仅用于
日志与审计，不是生成语音前的阻塞中间步骤。

## 两条生产路线

### A. 级联流式链路：最适合机器人工具控制

```text
Pi microphone
 → streaming ASR
 → LLM token/tool stream
 → Guidebot tool validation
 → bi-streaming expressive TTS
 → Pi speaker
```

优势是文字轨迹清楚、工具调用可审计、每个模型可替换。中文自托管建议在局域网 GPU 服务器上
运行 FunASR streaming + LLM + CosyVoice 3；CosyVoice 3 支持 text-in/audio-out 双向流式、
中文方言及情绪/语速指令。树莓派只负责音频 I/O，不承担大模型推理。

### B. 原生 speech-to-speech：最适合自然陪伴

原生实时音频模型直接消费用户音频并生成音频，能保留语调、停顿、笑声和情绪线索。可通过
`NativeSpeechSession` 接入 OpenAI Realtime、Gemini Live、Hume EVI 或自建 Qwen3-Omni 服务。

即使走原生音频，移动/灯光/舵机仍必须以结构化 tool call 返回 Guidebot；模型不能直接访问
I2C。`stop` 应在树莓派本地关键词模块执行，网络断开时也可停车。

## 推荐小车拓扑

```text
┌──────────────── Raspberry Pi / 小车 ────────────────┐
│ arecord + AEC + ring buffer                         │
│ /dev/myspeech wake/STOP fallback                    │
│ Guidebot Hub + SafetyPolicy + RaspbotDeviceAdapter  │
│ aplay + motor watchdog                              │
└───────────────────────┬─────────────────────────────┘
                        │ WebSocket/WebRTC, TLS
┌───────────────────────▼─────────────────────────────┐
│ Cloud or LAN GPU voice service                      │
│ streaming ASR/LLM/TTS or native speech-to-speech    │
└─────────────────────────────────────────────────────┘
```

不要在树莓派 4/5 上直接部署 Moshi 7B 或 Qwen3-Omni 30B。Moshi 官方 PyTorch 路径需要高显存
GPU；Qwen3-Omni 的 Talker 也需要大量显存。它们应部署在局域网 GPU 服务器或使用托管 API。

## 情绪价值不是单一模型开关

Guidebot 将其拆成四层：

1. **感知**：从音高、能量、语速、停顿和文本获取 `AffectState`，保留 confidence。
2. **对话策略**：先确认用户意图和感受，控制回应长度；低置信时不武断贴情绪标签。
3. **长期关系**：记住用户明确表达的称呼、偏好与边界，不把瞬时情绪永久化。
4. **声音表达**：将 `ProsodyStyle` 映射到 TTS 的 warmth、energy、rate 或自然语言 instruct。

`EmpathicStylePolicy` 已提供确定性、可测试的情绪到韵律策略。真正的情绪检测和 expressive TTS
由 provider adapter 实现。医疗、危机或强烈负面情绪不能仅靠“共情话术”，应进入专门安全策略。

## 推荐选择

| 目标 | 推荐路径 | 原因 |
|---|---|---|
| 最快得到自然中文对话 | Gemini Live affective / Qwen3-Omni API | 原生音频、低延迟、中文支持 |
| 机器人动作可靠优先 | streaming ASR + LLM + CosyVoice 3 | 工具与文本轨迹可审计 |
| 情绪理解优先 | Hume EVI 或 Gemini affective dialog | 原生分析语音表达信号 |
| 全离线研究 | LAN GPU 上 Qwen3-Omni 或 Moshi | 不依赖云，但硬件成本高 |
| 低成本离线保底 | `/dev/myspeech` 固定命令 | 可唤醒、停止、基础动作 |

Sesame CSM 是高质量 conversational TTS，而不是通用 LLM，且官方说明非英语效果有限；不建议
作为中文 Guidebot 的第一选择。

## 上车步骤

1. 硬件检查：`arecord -l`、`aplay -l`、`ls -l /dev/myspeech`。
2. 安装 Guidebot 与 `voice-hardware` 可选依赖。
3. 用 `guidebot voice-demo` 验证编排，再做麦克风回环测试。
4. 配置系统级 AEC；没有 AEC 时，扬声器声音会被麦克风当成用户插话。
5. 选择并实现一个 provider adapter，密钥只放环境变量或 secret manager。
6. 接入 `RaspbotDeviceAdapter`、本地 STOP 和运动 watchdog 后，才开放语音运动工具。
7. 记录首音频延迟、误打断率、ASR 字错率、工具成功率和用户主观自然度。

建议验收目标：首音频 P50 < 800 ms、P95 < 1.5 s；停止命令本地响应 < 200 ms；任何网络或
provider 异常都触发音频取消与底盘安全停止。
