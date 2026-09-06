import json

from guidebot.cli import main


def _json_from_stdout(capsys):
    return json.loads(capsys.readouterr().out)


def test_intent_parse_alarm_json(capsys) -> None:
    main(["intent", "parse", "明早七点叫我", "--json"])

    payload = _json_from_stdout(capsys)
    assert payload["intent_type"] == "set_alarm"


def test_alarm_set_preserves_machine_time_hint(capsys, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    main(["alarm", "set", "--time", "+5s", "--json"])

    payload = _json_from_stdout(capsys)
    assert payload["intent"]["slots"]["time"] == "+5s"
    assert payload["action"]["time"] == "+5s"


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


def test_ad_demo_json(capsys, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    main(["ad", "demo", "--product", "Guidebot", "--json"])

    payload = _json_from_stdout(capsys)
    assert payload["task"]["skill_id"] == "ad.generate_brief"
    assert payload["action"]["requires_human_review"] is True


def test_eval_run_json(capsys) -> None:
    main(["eval", "run", "--json"])

    payload = _json_from_stdout(capsys)
    assert payload["score"] == 1.0
    assert payload["failed"] == 0


def test_tool_contract_can_be_inspected(capsys) -> None:
    main(["tools", "describe", "ad.generate_brief", "--json"])

    payload = _json_from_stdout(capsys)
    assert payload["contract"]["permission"] == "read_only"
    assert payload["contract"]["timeout_s"] == 30.0


def test_high_level_planner_demo_uses_fixed_low_level_speed(capsys) -> None:
    main(["planner", "demo", "--scenario", "approach", "--json"])

    payload = _json_from_stdout(capsys)
    assert payload["high_level_plan"]["steps"][0]["option"] == "move_closer"
    action = payload["trajectory"]["accepted_actions"][0]
    assert action["kind"] == "move"
    assert action["parameters"]["speed"] == 0.25


def test_break_reminder_cli_runs_complete_trajectory(capsys) -> None:
    main(["demo", "break-reminder"])

    payload = _json_from_stdout(capsys)
    assert payload["status"] == "finished"
    assert payload["trajectory"][3]["tool_result"]["data"]["obstacle"] is True


def test_replay_cli_verifies_fire_twice(capsys) -> None:
    main(["replay", "data/replays/fire_verify.json"])

    payload = _json_from_stdout(capsys)
    tools = [step["decision"]["tool_name"] for step in payload["trajectory"]]
    assert tools == ["scene_inspect", "scene_inspect", "speak", None]
