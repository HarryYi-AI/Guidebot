# Raspberry Pi adapter examples

Copy these files to the robot, for example:

```bash
mkdir -p /home/pi/project_demo/guidebot_adapters
cp examples/raspberry_pi_adapters/*.py /home/pi/project_demo/guidebot_adapters/
```

They are intentionally thin wrappers. The original Yahboom/course source stays
under `/home/pi/project_demo`; Guidebot only consumes the JSON events printed by
these adapters.

Current adapters:

- `legacy_scene_monitor.py`: camera → DashScope-compatible VLM → scene JSON.
- `ultrasonic_stream.py`: Yahboom ultrasonic sensor → JSONL obstacle stream.
- `spin_alarm_ultrasonic_stop.py`: opt-in hardware alarm action used through `GUIDEBOT_ALARM_COMMAND`.

Useful commands:

```bash
python3 /home/pi/project_demo/guidebot_adapters/legacy_scene_monitor.py --guidebot-json --no-speak
python3 /home/pi/project_demo/guidebot_adapters/ultrasonic_stream.py
python3 /home/pi/project_demo/guidebot_adapters/spin_alarm_ultrasonic_stop.py --now --speed 25
```
