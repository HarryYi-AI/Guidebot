"""Shared terminal outcome semantics for runtime trajectories."""

from enum import Enum


class OutcomeType(str, Enum):
    EXECUTED = "executed"
    SUPPRESSED = "suppressed"
    NO_ACTION_REQUIRED = "no_action_required"
    FAILED = "failed"
