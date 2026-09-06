from guidebot.model_serving import (
    DeterministicBackend,
    InstrumentedBackend,
    ModelRequest,
    benchmark_backend,
)


def test_instrumented_backend_records_success_and_tokens() -> None:
    backend = InstrumentedBackend(DeterministicBackend())

    response = backend.infer(ModelRequest("ad_creative_brief", {"product": "Guidebot"}))
    metrics = backend.snapshot()

    assert response.text is not None
    assert metrics.request_count == 1
    assert metrics.success_count == 1
    assert metrics.error_count == 0
    assert metrics.total_output_tokens > 0


def test_benchmark_reports_latency_and_throughput() -> None:
    report = benchmark_backend(
        DeterministicBackend(),
        ModelRequest("benchmark", {"prompt": "hello"}),
        iterations=3,
    )

    assert report.requests == 3
    assert report.successes == 3
    assert report.errors == 0
    assert report.requests_per_second > 0
    assert report.p95_latency_ms >= report.p50_latency_ms
