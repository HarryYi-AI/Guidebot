# AutoSceneMonitor

这个目录是独立新增的自动场景巡检功能，不修改原来的 `SceneDescription` 代码。

## 功能流程

1. 每隔一段时间自动拍摄当前环境。
2. 把图片发送给通义视觉大模型，生成场景描述。
3. 把场景描述发送给同一个视觉大模型，结合异常规则判断是否存在异常。
4. 如果发现异常，在控制台输出反馈，并默认调用原项目的 TTS 语音播报。
5. 每次巡检结果写入 `AutoSceneMonitor/logs/monitor_log.jsonl`，图片保存在 `AutoSceneMonitor/captures/`。
6. `captures` 目录默认最多保存 5GB 照片，超过后会按最旧照片优先删除。

## 运行方式

在项目根目录运行：

```powershell
python .\AutoSceneMonitor\auto_scene_monitor.py --interval 300
```

只测试一次：

```powershell
python .\AutoSceneMonitor\auto_scene_monitor.py --once
```

只测试语音播报，不拍照、不调用视觉模型：

```powershell
python .\AutoSceneMonitor\auto_scene_monitor.py --test-speak
```

发现异常时默认会语音播报。如果调试时只想打印文字、不播报，可以关闭语音：

```powershell
python .\AutoSceneMonitor\auto_scene_monitor.py --interval 300 --no-speak
```

如果摄像头不是默认编号：

```powershell
python .\AutoSceneMonitor\auto_scene_monitor.py --camera 1 --once
```

修改照片保存容量上限：

```powershell
python .\AutoSceneMonitor\auto_scene_monitor.py --interval 300 --max-capture-gb 5
```

不自动清理照片：

```powershell
python .\AutoSceneMonitor\auto_scene_monitor.py --interval 300 --max-capture-gb 0
```

## 异常规则

默认会读取 `AutoSceneMonitor/watch_rule.txt`，直接运行时不需要再额外指定规则文件。

可以写一个 UTF-8 文本文件，例如 `watch_rule.txt`：

```text
重点检查实验室场景：明火、烟雾、人员摔倒、门未关闭、设备指示灯异常、地面明显积水。
普通人员经过和正常坐姿不算异常。
```

如果要换成别的规则文件，可以运行时指定：

```powershell
python .\AutoSceneMonitor\auto_scene_monitor.py --watch-rule-file .\AutoSceneMonitor\watch_rule.txt
```

## 图片测试

如果没有真实异常场景，可以下载公开图片模拟摄像头拍摄结果：

```powershell
python .\AutoSceneMonitor\download_test_images.py
```

下载完成后运行图片测试：

```powershell
python .\AutoSceneMonitor\test_with_images.py
```

当前测试图片覆盖：

```text
normal_office：正常办公室
window_closed：窗户关闭对照
window_open：窗户打开
fire_or_smoke：火焰/烟雾
flooded_room：室内积水
person_on_floor：人员倒地
blocked_fire_exit：消防出口/通道被堵
overloaded_socket：插座过载/冒烟
unmaintained_fire_equipment：消防设备异常
broken_window_glass：窗户破损/玻璃碎裂
camera_night_vision_issue：摄像头画面过暗/反光/遮挡异常
```

图片测试发现异常时也会默认语音播报。如果只想看文字结果：

```powershell
python .\AutoSceneMonitor\test_with_images.py --no-speak
```

测试结果会保存到：

```text
AutoSceneMonitor/logs/image_test_results.jsonl
```

## API Key 配置

推荐把这个实验用到的 API Key 配置在 `AutoSceneMonitor/local_config.py` 里，这样把整个 `AutoSceneMonitor` 文件夹复制到树莓派时，配置会一起带过去。

脚本会优先读取：

```text
AutoSceneMonitor/local_config.py
```

如果这个文件没有配置有效 key，才会回退读取项目根目录的：

```text
API_KEY.py
```

`local_config.py` 里主要配置：

- `TONYI_key`：通义千问 API Key；
- `TTS_IAT_Tongyi`：决定 TTS 播报走通义还是讯飞。

示例：

```python
TONYI_key = "你的新通义API_KEY"
TTS_IAT_Tongyi = True
TONYI_API_TTS_MODEL = "qwen-tts"
TONYI_TTS_VOICE = "Cherry"
```

如果原来的 `SceneDescription` 功能已经能在树莓派上运行，这个新实验不需要再单独安装一套依赖。

默认模型：

- 视觉描述：`qwen-vl-plus`
- 异常判断：默认与视觉描述相同，也是 `qwen-vl-plus`

可以用参数替换：

```powershell
python .\AutoSceneMonitor\auto_scene_monitor.py --vision-model qwen-vl-plus
```

如果确实想让异常判断使用另一个模型，也可以额外指定：

```powershell
python .\AutoSceneMonitor\auto_scene_monitor.py --vision-model qwen-vl-plus --judge-model qwen-plus
```
