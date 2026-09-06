"""Composable agent roles for Guidebot.

These roles are optional orchestration components. Runtime hardware dispatch
still flows through EventBus, IntentAnalyzer, RuntimeSkillRegistry, and
SafetyGate.
"""

from .embodied_planner import EmbodiedPlannerAgent, PlannerClient, ScriptedPlannerClient
from .skill_evolution import SkillEvolutionAgent, SkillEvolutionReport
from guidebot.planning import FixedOptionCompiler, HighLevelPlan, OptionStep, SkillOption

__all__ = [
    "EmbodiedPlannerAgent",
    "FixedOptionCompiler",
    "HighLevelPlan",
    "OptionStep",
    "PlannerClient",
    "ScriptedPlannerClient",
    "SkillEvolutionAgent",
    "SkillEvolutionReport",
    "SkillOption",
]
