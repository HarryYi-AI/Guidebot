"""Repeatable latency and throughput benchmark for any ModelBackend."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from statistics import mean
from time import perf_counter

from .backend import ModelBackend
from .models import ModelRequest


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    backend: str
    requests: int
    successes: int
    errors: int
    wall_time_s: float
    requests_per_second: float
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    output_tokens_per_second: float | None


def benchmark_backend(
    backend: ModelBackend,
    request: ModelRequest,
    *,
    iterations: int = 10,
) -> BenchmarkReport:
    if iterations < 1:
        raise ValueError("iterations must be at least one")
    latencies: list[float] = []
    output_tokens = 0
    errors = 0
    started = perf_counter()
    for _ in range(iterations):
        request_started = perf_counter()
        try:
            response = backend.infer(request)
        except Exception:  # noqa: BLE001 - benchmark counts provider-specific failures.
            errors += 1
            continue
        latencies.append(response.latency_ms or (perf_counter() - request_started) * 1_000)
        output_tokens += response.output_tokens or 0
    wall_time_s = max(perf_counter() - started, 1e-9)
    ordered = sorted(latencies)
    successes = len(ordered)
    return BenchmarkReport(
        backend=backend.name,
        requests=iterations,
        successes=successes,
        errors=errors,
        wall_time_s=wall_time_s,
        requests_per_second=successes / wall_time_s,
        mean_latency_ms=mean(ordered) if ordered else 0.0,
        p50_latency_ms=_percentile(ordered, 0.50),
        p95_latency_ms=_percentile(ordered, 0.95),
        output_tokens_per_second=(output_tokens / wall_time_s) if output_tokens else None,
    )


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    index = max(0, ceil(len(values) * quantile) - 1)
    return values[index]
