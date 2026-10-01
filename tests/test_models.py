from agent_tps_bench.models import (
    BenchmarkResult,
    BenchmarkStatus,
    TimeoutType,
    TimingMetrics,
    TokenMetrics,
    TPSMetrics,
)


def test_token_metrics_properties():
    tok = TokenMetrics(
        input_tokens=100,
        output_tokens=50,
        reasoning_tokens=25,
        cached_read_tokens=400,
        cached_write_tokens=10,
    )
    assert tok.generated_tokens == 75  # 50 output + 25 reasoning
    assert tok.total_tokens == 100 + 50 + 25 + 10  # 185
    # total input = 100 + 400 = 500. cache hit rate = 400 / 500 = 0.8
    assert abs(tok.cache_hit_rate - 0.8) < 1e-4


def test_timing_and_tps_models():
    timings = TimingMetrics(
        request_start_ms=1000.0,
        first_token_ms=1200.0,
        completed_ms=2200.0,
        ttft_ms=200.0,
        generation_duration_ms=1000.0,
        total_duration_ms=1200.0,
    )
    tps = TPSMetrics(decode_tps=50.0, e2e_tps=41.67, total_throughput_tps=100.0)
    res = BenchmarkResult(
        id="test-1",
        provider="test-prov",
        model="test-model",
        status=BenchmarkStatus.SUCCESS,
        timeout_type=TimeoutType.NONE,
        timings=timings,
        tps=tps,
    )
    assert res.status == BenchmarkStatus.SUCCESS
    assert res.tps.decode_tps == 50.0
    assert res.timings.ttft_ms == 200.0
