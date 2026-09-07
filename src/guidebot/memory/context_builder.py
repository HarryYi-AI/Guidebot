"""Reconstruct a typed, minimal memory context from a query plan."""

from __future__ import annotations

from .retriever import StructuredMemoryRetriever
from .schemas import MemoryContext, MemoryQueryPlan


class MemoryContextBuilder:
    def __init__(self, retriever: StructuredMemoryRetriever) -> None:
        self.retriever = retriever

    def build(self, user_id: str, plan: MemoryQueryPlan) -> MemoryContext:
        preferences = (
            self.retriever.retrieve_preferences(user_id, plan.preference_keys)
            if plan.preference_keys
            else ()
        )
        core = self.retriever.retrieve_core_memory(user_id) if plan.need_core else ()
        # A relevant preference belongs in the dedicated field, not twice in the prompt context.
        core = tuple(item for item in core if item not in preferences)
        return MemoryContext(
            core=core,
            current_state=self.retriever.retrieve_active_states(user_id)
            if plan.need_current_state
            else (),
            facts=self.retriever.retrieve_facts(user_id, plan.fact_keys),
            preferences=preferences,
            episodes=self.retriever.retrieve_recent_episodes(
                user_id,
                plan.query,
                limit=plan.episode_limit,
            )
            if plan.need_recent_episodes
            else (),
            boundaries=self.retriever.retrieve_boundaries(user_id, plan)
            if plan.need_core or plan.boundary_keys
            else (),
            skill_evidence=self.retriever.retrieve_skill_evidence(user_id, plan.query)
            if plan.need_skill_evidence
            else (),
        )
