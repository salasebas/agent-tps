from agent_tps.core.calculator import (
    aggregate_concurrency_results,
    calculate_jitter,
    calculate_percentile,
    compute_timing_metrics,
    compute_tps_metrics,
)
from agent_tps.core.models import (
    BenchmarkResult,
    BenchmarkStatus,
    TimeoutType,
    TimingMetrics,
    TokenMetrics,
)


def test_percentile_calculation():
    data = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert calculate_percentile(data, 50.0) == 30.0
    assert calculate_percentile(data, 0.0) == 10.0
    assert calculate_percentile(data, 100.0) == 50.0
    assert calculate_percentile([], 50.0) == 0.0


def test_jitter_calculation():
    # Identical delays -> 0 jitter
    assert calculate_jitter([10.0, 10.0, 10.0]) == 0.0
    # Alternating delays
    assert calculate_jitter([10.0, 20.0, 10.0]) == 10.0
    assert calculate_jitter([5.0]) == 0.0


def test_compute_timing_metrics():
    latencies = [10.0, 15.0, 20.0, 25.0]
    timings = compute_timing_metrics(
        request_start_ms=1000.0,
        first_token_ms=1300.0,
        completed_ms=2000.0,
        inter_token_latencies_ms=latencies,
        connection_ms=1050.0,
    )
    assert timings.ttft_ms == 300.0
    assert timings.generation_duration_ms == 700.0
    assert timings.total_duration_ms == 1000.0
    assert timings.itl_p50_ms == 17.5
    assert timings.itl_max_ms == 25.0
    assert timings.jitter_ms == 5.0


def test_compute_tps_metrics():
    tokens = TokenMetrics(input_tokens=100, output_tokens=80, reasoning_tokens=20)
    timings = TimingMetrics(
        request_start_ms=1000.0,
        first_token_ms=1200.0,
        completed_ms=2200.0,
        generation_duration_ms=1000.0,  # 1.0 second
        total_duration_ms=1200.0,  # 1.2 seconds
    )
    tps = compute_tps_metrics(tokens, timings)
    # generated = 80 + 20 = 100 tokens. In 1.0s gen time -> decode_tps = 100.0
    assert tps.decode_tps == 100.0
    # in 1.2s total time -> e2e_tps = 100 / 1.2 = 83.33
    assert tps.e2e_tps == 83.33
    # total tokens = 200. total throughput = 200 / 1.2 = 166.67
    assert tps.total_throughput_tps == 166.67


def test_aggregate_concurrency_results():
    results = [
        BenchmarkResult(
            id="1",
            provider="test",
            model="m1",
            status=BenchmarkStatus.SUCCESS,
            tokens=TokenMetrics(output_tokens=100),
            timings=TimingMetrics(
                request_start_ms=0,
                first_token_ms=200,
                completed_ms=1200,
                ttft_ms=200,
                generation_duration_ms=1000,
                total_duration_ms=1200,
            ),
            tps=compute_tps_metrics(
                TokenMetrics(output_tokens=100),
                TimingMetrics(
                    request_start_ms=0,
                    first_token_ms=200,
                    completed_ms=1200,
                    generation_duration_ms=1000,
                    total_duration_ms=1200,
                ),
            ),
        ),
        BenchmarkResult(
            id="2",
            provider="test",
            model="m1",
            status=BenchmarkStatus.RATE_LIMITED,
            timeout_type=TimeoutType.RATE_LIMIT_429,
            timings=TimingMetrics(request_start_ms=0, completed_ms=100),
        ),
    ]
    report = aggregate_concurrency_results(
        concurrency_level=2,
        results=results,
        wall_clock_duration_s=1.2,
        baseline_single_worker_tps=120.0,
    )
    assert report.concurrency_level == 2
    assert report.total_requests == 2
    assert report.successful_requests == 1
    assert report.rate_limited_requests == 1
    assert report.aggregate_decode_tps == round(100 / 1.2, 2)
    assert report.degradation_percent > 0.0
