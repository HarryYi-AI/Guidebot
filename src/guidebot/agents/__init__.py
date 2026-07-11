"""Composable agent roles for Guidebot.

These roles are optional orchestration components. Runtime hardware dispatch
still flows through EventBus, IntentAnalyzer, RuntimeSkillRegistry, and
SafetyGate.
"""

from .embodied_planner import EmbodiedPlannerAgent, PlannerClient, ScriptedPlannerClient
from .skill_evolution import SkillEvolutionAgent, SkillEvolutionReport

__all__ = [
    "EmbodiedPlannerAgent",
    "PlannerClient",
    "ScriptedPlannerClient",
    "SkillEvolutionAgent",
    "SkillEvolutionReport",
]
