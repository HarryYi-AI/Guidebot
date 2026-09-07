from datetime import timedelta

import pytest

from guidebot.events import Event
from guidebot.memory import (
    BoundaryMemory,
    EpisodeMemory,
    FactMemory,
    MemoryService,
    MemoryStatus,
    MockLLMMemoryExtractor,
    PreferenceMemory,
    ResolutionAction,
    RuleBasedMemoryQueryPlanner,
    SkillEvidenceMemory,
    StructuredMemoryStore,
    TemporaryStateMemory,
)
from guidebot.models import utc_now


def _service(tmp_path) -> MemoryService:
    return MemoryService(StructuredMemoryStore(tmp_path / "memory.sqlite3"))


def _climate_feedback(service: MemoryService, user: str, temperature: float, success: bool) -> str:
    event = Event(
        "climate.feedback",
        "voice",
        {
            "text": "舒适" if success else "太冷",
            "period": "night",
            "target_temperature": temperature,
            "comfortable": success,
            "skill_name": "NightComfortSkill",
            "applicability": "night temperature comfort",
            "procedure": [f"set_{temperature}C", "ask_feedback"],
            "success": success,
            "outcome": "comfortable" if success else "too cold",
        },
    )
    episode = service.record_event(
        event,
        user_id=user,
        outcome={
            "period": "night",
            "target_temperature": temperature,
            "comfortable": success,
        },
    )
    return episode.memory_id


def test_tired_is_temporary_ttl_state_not_preference(tmp_path) -> None:
    service = _service(tmp_path)
    now = utc_now()
    service.record_event(
        Event("user.text", "voice", {"text": "我今天有点累"}, timestamp=now),
        user_id="u",
    )

    states = service.store.list(TemporaryStateMemory, "u", status=MemoryStatus.ACTIVE)
    assert states[0].state == "tired"
    assert states[0].expires_at > states[0].valid_from
    assert service.store.list(PreferenceMemory, "u") == ()

    service.retriever.retrieve_active_states("u", now=now + timedelta(hours=5))
    assert service.store.list(TemporaryStateMemory, "u", status=MemoryStatus.ACTIVE) == ()
    assert service.store.list(TemporaryStateMemory, "u", status=MemoryStatus.EXPIRED)
    assert len(service.store.list(EpisodeMemory, "u")) == 1


def test_repeated_night_comfort_consolidates_preference_and_skill_candidate(tmp_path) -> None:
    service = _service(tmp_path)
    _climate_feedback(service, "u", 23.0, False)
    first = _climate_feedback(service, "u", 25.0, True)
    second = _climate_feedback(service, "u", 25.0, True)

    report = service.run_consolidation("u")

    preference = service.store.list(PreferenceMemory, "u", status=MemoryStatus.ACTIVE)[0]
    assert preference.value["preferred_c"] == 25.0
    assert set(preference.evidence_episode_ids) == {first, second}
    assert report.contradictions
    assert report.skill_candidates[0].accepted is False
    assert report.skill_candidates[0].success_count == 2


def test_conflicting_preference_historizes_old_value(tmp_path) -> None:
    service = _service(tmp_path)
    old = PreferenceMemory("u", "night_temperature_preference", 23, 0.8, ("e1",))
    new = PreferenceMemory("u", "night_temperature_preference", 25, 0.9, ("e2",))
    service.update_memory(old, actor="consolidator")

    result = service.update_memory(new, actor="consolidator")

    assert result.action is ResolutionAction.HISTORIZE
    assert service.store.list(PreferenceMemory, "u", status=MemoryStatus.ACTIVE) == (new,)
    historical = service.store.list(PreferenceMemory, "u", status=MemoryStatus.HISTORICAL)
    assert historical[0].value == 23
    assert historical[0].valid_to == new.valid_from


def test_agent_cannot_overwrite_boundary(tmp_path) -> None:
    service = _service(tmp_path)
    original = BoundaryMemory("u", "physical_action_safety", "all movement uses safety gate")
    service.update_memory(original, actor="system")

    with pytest.raises(PermissionError):
        service.update_memory(
            BoundaryMemory("u", "physical_action_safety", "skip safety gate"),
            actor="agent",
        )

    assert service.store.list(BoundaryMemory, "u", status=MemoryStatus.ACTIVE) == (original,)


def test_single_message_cannot_directly_write_stable_preference(tmp_path) -> None:
    service = _service(tmp_path)

    with pytest.raises(PermissionError):
        service.update_memory(PreferenceMemory("u", "music", "jazz", 0.8, ("one",)))


def test_cold_night_query_requests_minimal_relevant_memory() -> None:
    plan = RuleBasedMemoryQueryPlanner().plan_memory_query("今晚有点冷")

    assert plan.need_current_state is True
    assert plan.need_recent_episodes is True
    assert plan.need_skill_evidence is True
    assert "night_temperature_preference" in plan.preference_keys
    assert "physical_action_safety" in plan.boundary_keys


def test_query_reconstructs_typed_context(tmp_path) -> None:
    service = _service(tmp_path)
    service.update_memory(
        BoundaryMemory("u", "physical_action_safety", "all movement uses safety gate"),
        actor="system",
    )
    _climate_feedback(service, "u", 25.0, True)
    _climate_feedback(service, "u", 25.0, True)
    service.run_consolidation("u")

    context = service.get_context("u", "今晚有点冷")

    assert context.preferences[0].key == "night_temperature_preference"
    assert context.boundaries[0].key == "physical_action_safety"
    assert context.episodes
    assert context.skill_evidence


def test_retrieval_recalls_relevant_temperature_skill_evidence(tmp_path) -> None:
    service = _service(tmp_path)
    cold_id = _climate_feedback(service, "u", 23.0, False)
    warm_id = _climate_feedback(service, "u", 24.5, True)

    evidence = service.retriever.retrieve_skill_evidence("u", "夜间温度舒适")

    assert {item.source_episode_id for item in evidence} == {cold_id, warm_id}
    assert all(isinstance(item, SkillEvidenceMemory) for item in evidence)


def test_single_negative_emotion_does_not_change_stable_profile(tmp_path) -> None:
    service = _service(tmp_path)
    service.record_event(
        Event("user.text", "voice", {"text": "我今天有点焦虑"}),
        user_id="u",
    )
    service.run_consolidation("u")

    assert service.store.list(PreferenceMemory, "u") == ()
    assert service.store.list(FactMemory, "u") == ()


def test_mock_llm_extractor_enforces_strict_json_schema() -> None:
    extractor = MockLLMMemoryExtractor(
        '{"candidates":[{"type":"fact","key":"user_name",'
        '"value":"小明","confidence":0.9}]}'
    )
    event = Event("user.text", "voice", {"text": "我叫小明"})

    candidates = extractor.extract_memory_candidates(
        event,
        user_id="u",
        source_episode_id="episode",
    )

    assert isinstance(candidates[0], FactMemory)
    with pytest.raises(ValueError):
        MockLLMMemoryExtractor('{"unexpected":[]}').extract_memory_candidates(
            event,
            user_id="u",
            source_episode_id="episode",
        )
