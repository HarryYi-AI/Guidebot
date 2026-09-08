from guidebot.memory import (
    ACTIVE_TASK_PATH,
    CORE_MEMORY,
    CURRENT_STATE,
    RECENT_OBSERVATIONS,
    RETRIEVED_HISTORY,
    SAFETY,
    ContextBudgetAllocator,
)


def test_context_budget_respects_caps_and_keeps_recent_observations() -> None:
    allocator = ContextBudgetAllocator(total_tokens=100)
    parts = {
        CORE_MEMORY: "用户稳定偏好" * 30,
        CURRENT_STATE: [f"state-{index}" for index in range(20)],
        ACTIVE_TASK_PATH: [f"task-{index}" for index in range(20)],
        RECENT_OBSERVATIONS: [f"observation-{index}" for index in range(20)],
        RETRIEVED_HISTORY: [f"episode-{index}" for index in range(20)],
        SAFETY: "never bypass safety gate" * 10,
    }

    allocated = allocator.allocate(parts)

    assert set(allocated) == set(parts)
    assert all(usage.used <= usage.cap for usage in allocator.last_usage.values())
    assert sum(usage.cap for usage in allocator.last_usage.values()) <= 100
    assert allocated[RECENT_OBSERVATIONS][-1] == "observation-19"
    assert allocator.last_usage[SAFETY].cap == 5
    assert all(usage.truncated for usage in allocator.last_usage.values())


def test_context_budget_supports_per_section_max_cap() -> None:
    allocator = ContextBudgetAllocator(total_tokens=200, max_caps={ACTIVE_TASK_PATH: 12})

    allocator.allocate({ACTIVE_TASK_PATH: "x" * 1_000})

    assert allocator.last_usage[ACTIVE_TASK_PATH].cap == 12
    assert allocator.last_usage[ACTIVE_TASK_PATH].used <= 12


def test_active_task_budget_preserves_root_and_current_node() -> None:
    allocator = ContextBudgetAllocator(total_tokens=100)
    path = [
        {"id": "root", "goal": "r" * 100},
        {"id": "middle", "goal": "m" * 100},
        {"id": "current", "goal": "c" * 100},
    ]

    allocated = allocator.allocate({ACTIVE_TASK_PATH: path})[ACTIVE_TASK_PATH]

    assert [node["id"] for node in allocated] == ["root", "current"]
    assert allocator.last_usage[ACTIVE_TASK_PATH].used <= 30
