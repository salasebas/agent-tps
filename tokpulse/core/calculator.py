from __future__ import annotations

from collections.abc import Sequence
import math

from tokpulse.core.models import (
    BenchmarkResult,
    BenchmarkStatus,
    ConcurrencyReport,
    TimeoutType,
    TimingMetrics,
    TokenMetrics,
    TPSMetrics,
)


def calculate_percentile(data: Sequence[float], percentile: float) -> float:
    """Calculate the p-th percentile of a sequence using linear interpolation."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n == 1:
        return sorted_data[0]

    k = (n - 1) * (percentile / 100.0)
    floor_k = math.floor(k)
    ceil_k = math.ceil(k)

    if floor_k == ceil_k:
        return sorted_data[int(k)]

    d0 = sorted_data[floor_k] * (ceil_k - k)
    d1 = sorted_data[ceil_k] * (k - floor_k)
    return d0 + d1


def calculate_jitter(latencies: Sequence[float]) -> float:
    """Calculate inter-arrival jitter (RFC 3550 style).

    Represents statistical variance in inter-token arrival time.
    """
    if len(latencies) < 2:
        return 0.0
    diffs = [abs(latencies[i] - latencies[i - 1]) for i in range(1, len(latencies))]
    return sum(diffs) / len(diffs)


def compute_timing_metrics(
    request_start_ms: float,
    first_token_ms: float | None = None,
    completed_ms: float | None = None,
    inter_token_latencies_ms: list[float] | None = None,
    connection_ms: float | None = None,
) -> TimingMetrics:
    """Calculates all timing and latency percentile metrics."""
    latencies = inter_token_latencies_ms or []

    ttft_ms = None
    if first_token_ms is not None and first_token_ms >= request_start_ms:
        ttft_ms = first_token_ms - request_start_ms

    generation_duration_ms = None
    if first_token_ms is not None and completed_ms is not None and completed_ms >= first_token_ms:
        generation_duration_ms = completed_ms - first_token_ms

    total_duration_ms = None
    if completed_ms is not None and completed_ms >= request_start_ms:
        total_duration_ms = completed_ms - request_start_ms

    p50 = calculate_percentile(latencies, 50.0)
    p90 = calculate_percentile(latencies, 90.0)
    p95 = calculate_percentile(latencies, 95.0)
    p99 = calculate_percentile(latencies, 99.0)
    max_itl = max(latencies) if latencies else 0.0
    jitter = calculate_jitter(latencies)

    return TimingMetrics(
        request_start_ms=request_start_ms,
        connection_ms=connection_ms,
        first_token_ms=first_token_ms,
        completed_ms=completed_ms,
        ttft_ms=ttft_ms,
        generation_duration_ms=generation_duration_ms,
        total_duration_ms=total_duration_ms,
        inter_token_latencies_ms=latencies,
        itl_p50_ms=p50,
        itl_p90_ms=p90,
        itl_p95_ms=p95,
        itl_p99_ms=p99,
        itl_max_ms=max_itl,
        jitter_ms=jitter,
    )


def compute_tps_metrics(tokens: TokenMetrics, timings: TimingMetrics) -> TPSMetrics:
    """Compute decode TPS, end-to-end TPS, and total throughput TPS."""
    generated = tokens.generated_tokens

    # Decode TPS: tokens generated during active generation phase
    decode_tps = 0.0
    if timings.generation_duration_ms and timings.generation_duration_ms > 0:
        decode_tps = (generated / timings.generation_duration_ms) * 1000.0

    # End-to-end TPS: tokens generated relative to entire request turn duration
    e2e_tps = 0.0
    if timings.total_duration_ms and timings.total_duration_ms > 0:
        e2e_tps = (generated / timings.total_duration_ms) * 1000.0

    # Total throughput TPS: total processed (prompt + completion) relative to turn duration
    total_tps = 0.0
    if timings.total_duration_ms and timings.total_duration_ms > 0:
        total_tps = (tokens.total_tokens / timings.total_duration_ms) * 1000.0

    return TPSMetrics(
        decode_tps=round(decode_tps, 2),
        e2e_tps=round(e2e_tps, 2),
        total_throughput_tps=round(total_tps, 2),
    )


def build_error_benchmark_result(
    id: str,
    provider: str,
    model: str,
    timeout_type: TimeoutType,
    error_message: str,
    request_start_ms: float,
    first_token_ms: float | None = None,
    tokens: TokenMetrics | None = None,
    retry_after_s: float | None = None,
) -> BenchmarkResult:
    """Convenience builder for failed/timed-out benchmarks."""
    now_ms = request_start_ms
    import time

    now_ms = time.time() * 1000.0

    status = (
        BenchmarkStatus.RATE_LIMITED
        if timeout_type == TimeoutType.RATE_LIMIT_429
        else BenchmarkStatus.TIMEOUT
        if timeout_type
        in (
            TimeoutType.CONNECT_TIMEOUT,
            TimeoutType.TTFT_TIMEOUT,
            TimeoutType.STALL_TIMEOUT,
            TimeoutType.DEADLINE_TIMEOUT,
        )
        else BenchmarkStatus.ERROR
    )

    timings = compute_timing_metrics(
        request_start_ms=request_start_ms,
        first_token_ms=first_token_ms,
        completed_ms=now_ms,
    )
    tokens_actual = tokens or TokenMetrics()
    tps = compute_tps_metrics(tokens_actual, timings)

    return BenchmarkResult(
        id=id,
        provider=provider,
        model=model,
        status=status,
        timeout_type=timeout_type,
        tokens=tokens_actual,
        timings=timings,
        tps=tps,
        error_message=error_message,
        retry_after_s=retry_after_s,
    )


def compute_concurrency_report(
    concurrency_level: int,
    results: list[BenchmarkResult],
    wall_clock_duration_s: float,
    baseline_decode_tps: float | None = None,
    baseline_single_worker_tps: float | None = None,
) -> ConcurrencyReport:
    """Aggregates multi-agent stress results and calculates TPS degradation."""
    baseline = baseline_decode_tps or baseline_single_worker_tps
    total = len(results)
    successes = [r for r in results if r.status == BenchmarkStatus.SUCCESS]
    timeouts = [r for r in results if r.status == BenchmarkStatus.TIMEOUT]
    rate_limits = [r for r in results if r.status == BenchmarkStatus.RATE_LIMITED]
    failures = [
        r
        for r in results
        if r.status not in (BenchmarkStatus.SUCCESS, BenchmarkStatus.TIMEOUT, BenchmarkStatus.RATE_LIMITED)
    ]

    total_generated_tokens = sum(r.tokens.generated_tokens for r in results)
    aggregate_decode_tps = (
        (total_generated_tokens / wall_clock_duration_s) if wall_clock_duration_s > 0 else 0.0
    )

    worker_decode_tps_list = [r.tps.decode_tps for r in successes if r.tps.decode_tps > 0]
    mean_worker_decode_tps = (
        sum(worker_decode_tps_list) / len(worker_decode_tps_list) if worker_decode_tps_list else 0.0
    )

    ttfts = [r.timings.ttft_ms for r in results if r.timings.ttft_ms is not None]
    mean_ttft = sum(ttfts) / len(ttfts) if ttfts else 0.0
    p50_ttft = calculate_percentile(ttfts, 50.0)
    p95_ttft = calculate_percentile(ttfts, 95.0)
    p99_ttft = calculate_percentile(ttfts, 99.0)

    degradation = 0.0
    if baseline and baseline > 0 and mean_worker_decode_tps > 0:
        degradation = max(0.0, ((baseline - mean_worker_decode_tps) / baseline) * 100.0)

    return ConcurrencyReport(
        concurrency_level=concurrency_level,
        total_requests=total,
        successful_requests=len(successes),
        failed_requests=len(failures),
        timed_out_requests=len(timeouts),
        rate_limited_requests=len(rate_limits),
        wall_clock_duration_s=round(wall_clock_duration_s, 2),
        aggregate_decode_tps=round(aggregate_decode_tps, 2),
        aggregate_e2e_tps=round(
            (total_generated_tokens / wall_clock_duration_s) if wall_clock_duration_s > 0 else 0.0,
            2,
        ),
        mean_ttft_ms=round(mean_ttft, 2),
        p50_ttft_ms=round(p50_ttft, 2),
        p95_ttft_ms=round(p95_ttft, 2),
        p99_ttft_ms=round(p99_ttft, 2),
        mean_worker_decode_tps=round(mean_worker_decode_tps, 2),
        degradation_percent=round(degradation, 2),
        results=results,
    )


aggregate_concurrency_results = compute_concurrency_report
