"""Dependency-light model serving interfaces used by Guidebot modules."""

from .backend import ModelBackend
from .benchmark import BenchmarkReport, benchmark_backend
from .instrumentation import InstrumentedBackend, ServingMetrics
from .models import ModelRequest, ModelResponse
from .openai_compatible import OpenAICompatibleBackend
from .stub import DeterministicBackend

__all__ = [
    "BenchmarkReport",
    "DeterministicBackend",
    "InstrumentedBackend",
    "ModelBackend",
    "ModelRequest",
    "ModelResponse",
    "OpenAICompatibleBackend",
    "ServingMetrics",
    "benchmark_backend",
]
