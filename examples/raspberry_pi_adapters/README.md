# Raspberry Pi adapter examples

Copy these files to the robot, for example:

```bash
mkdir -p /home/pi/project_demo/guidebot_adapters
cp examples/raspberry_pi_adapters/*.py /home/pi/project_demo/guidebot_adapters/
```

They are intentionally thin wrappers. The original Yahboom/course source stays
under `/home/pi/project_demo`; Guidebot only consumes the JSON events printed by
these adapters.
