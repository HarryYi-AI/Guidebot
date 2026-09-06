"""Small observability wrapper shared by every model backend."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from time import perf_counter

from .backend import ModelBackend
from .models import ModelRequest, ModelResponse


@dataclass(frozen=True, slots=True)
class ServingMetrics:
    request_count: int
    success_count: int
    error_count: int
    total_latency_ms: float
    total_output_tokens: int

    @property
    def success_rate(self) -> float:
        return self.success_count / self.request_count if self.request_count else 0.0

    @property
    def average_latency_ms(self) -> float:
        return self.total_latency_ms / self.success_count if self.success_count else 0.0


class InstrumentedBackend:
    """Thread-safe metrics decorator; it does not change provider behavior."""

    def __init__(self, backend: ModelBackend) -> None:
        self.backend = backend
        self.name = f"instrumented:{backend.name}"
        self._lock = Lock()
        self._requests = 0
        self._successes = 0
        self._errors = 0
        self._latency_ms = 0.0
        self._output_tokens = 0

    def infer(self, request: ModelRequest) -> ModelResponse:
        started = perf_counter()
        with self._lock:
            self._requests += 1
        try:
            response = self.backend.infer(request)
        except Exception:
            with self._lock:
                self._errors += 1
            raise
        elapsed_ms = response.latency_ms or (perf_counter() - started) * 1_000
        with self._lock:
            self._successes += 1
            self._latency_ms += elapsed_ms
            self._output_tokens += response.output_tokens or 0
        return response

    def snapshot(self) -> ServingMetrics:
        with self._lock:
            return ServingMetrics(
                self._requests,
                self._successes,
                self._errors,
                self._latency_ms,
                self._output_tokens,
            )
