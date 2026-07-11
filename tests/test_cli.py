import json

from guidebot.cli import main


def _json_from_stdout(capsys):
    return json.loads(capsys.readouterr().out)


def test_intent_parse_alarm_json(capsys) -> None:
    main(["intent", "parse", "明早七点叫我", "--json"])

    payload = _json_from_stdout(capsys)
    assert payload["intent_type"] == "set_alarm"


def test_scene_scan_fire_json(capsys, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    main(["scene", "scan", "--label", "fire", "--json"])

    payload = _json_from_stdout(capsys)
    assert payload["intent"]["intent_type"] == "safety_fire_alert"
    assert payload["task"]["target_module"] == "scene_monitor"


def test_health_check_sedentary_json(capsys, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    main(["health", "check", "--sedentary", "--json"])

    payload = _json_from_stdout(capsys)
    assert payload["intent"]["intent_type"] == "health_sedentary"


def test_climate_status_json(capsys, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    main(["climate", "status", "--json"])

    payload = _json_from_stdout(capsys)
    assert payload["real_control_enabled"] is False


def test_climate_check_json(capsys, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    main(["climate", "check", "--temperature-c", "29", "--humidity", "75", "--json"])

    payload = _json_from_stdout(capsys)
    assert payload["intent"]["intent_type"] == "climate_comfort"
    assert payload["task"]["skill_id"] == "climate.comfort"


def test_climate_check_ac_left_on_json(capsys, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    main(["climate", "check", "--ac-on", "--unoccupied", "--json"])

    payload = _json_from_stdout(capsys)
    assert payload["intent"]["intent_type"] == "ac_left_on_alert"
    assert payload["task"]["skill_id"] == "climate.ac_left_on"
