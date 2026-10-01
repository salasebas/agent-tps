from __future__ import annotations

import math
from typing import Sequence
from agent_tps_bench.models import (
    BenchmarkResult,
    BenchmarkStatus,
    ConcurrencyReport,
    TimingMetrics,
    TokenMetrics,
    TPSMetrics,
    TimeoutType,
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
        gen_seconds = timings.generation_duration_ms / 1000.0
        decode_tps = generated / gen_seconds

    # E2E TPS: tokens generated over the entire request lifecycle (including TTFT)
    e2e_tps = 0.0
    if timings.total_duration_ms and timings.total_duration_ms > 0:
        total_seconds = timings.total_duration_ms / 1000.0
        e2e_tps = generated / total_seconds

    # Total throughput TPS: all tokens (input + output + reasoning) / total request time
    total_throughput_tps = 0.0
    if timings.total_duration_ms and timings.total_duration_ms > 0:
        total_seconds = timings.total_duration_ms / 1000.0
        total_throughput_tps = tokens.total_tokens / total_seconds

    return TPSMetrics(
        decode_tps=round(decode_tps, 2),
        e2e_tps=round(e2e_tps, 2),
        total_throughput_tps=round(total_throughput_tps, 2),
    )


def aggregate_concurrency_results(
    concurrency_level: int,
    results: list[BenchmarkResult],
    wall_clock_duration_s: float,
    baseline_single_worker_tps: float | None = None,
) -> ConcurrencyReport:
    """Aggregates a set of benchmark results across concurrent workers."""
    total = len(results)
    successful = [r for r in results if r.status == BenchmarkStatus.SUCCESS]
    timed_out = [r for r in results if r.status == BenchmarkStatus.TIMEOUT]
    rate_limited = [r for r in results if r.status == BenchmarkStatus.RATE_LIMITED]
    failed = [r for r in results if r.status == BenchmarkStatus.ERROR]

    total_generated_tokens = sum(r.tokens.generated_tokens for r in successful)

    aggregate_decode_tps = (
        round(total_generated_tokens / wall_clock_duration_s, 2)
        if wall_clock_duration_s > 0
        else 0.0
    )

    all_ttfts = [r.timings.ttft_ms for r in successful if r.timings.ttft_ms is not None]
    mean_ttft = sum(all_ttfts) / len(all_ttfts) if all_ttfts else 0.0
    p50_ttft = calculate_percentile(all_ttfts, 50.0)
    p95_ttft = calculate_percentile(all_ttfts, 95.0)
    p99_ttft = calculate_percentile(all_ttfts, 99.0)

    worker_decode_tps_list = [r.tps.decode_tps for r in successful if r.tps.decode_tps > 0]
    mean_worker_decode_tps = (
        sum(worker_decode_tps_list) / len(worker_decode_tps_list)
        if worker_decode_tps_list
        else 0.0
    )

    worker_e2e_tps_list = [r.tps.e2e_tps for r in successful if r.tps.e2e_tps > 0]
    aggregate_e2e_tps = (
        round(sum(worker_e2e_tps_list), 2)
        if worker_e2e_tps_list
        else 0.0
    )

    # Degradation percent compared to single worker baseline
    degradation = 0.0
    if baseline_single_worker_tps and baseline_single_worker_tps > 0 and mean_worker_decode_tps > 0:
        drop = baseline_single_worker_tps - mean_worker_decode_tps
        degradation = round(max(0.0, (drop / baseline_single_worker_tps) * 100.0), 2)

    return ConcurrencyReport(
        concurrency_level=concurrency_level,
        total_requests=total,
        successful_requests=len(successful),
        failed_requests=len(failed),
        timed_out_requests=len(timed_out),
        rate_limited_requests=len(rate_limited),
        wall_clock_duration_s=round(wall_clock_duration_s, 3),
        aggregate_decode_tps=aggregate_decode_tps,
        aggregate_e2e_tps=aggregate_e2e_tps,
        mean_ttft_ms=round(mean_ttft, 2),
        p50_ttft_ms=round(p50_ttft, 2),
        p95_ttft_ms=round(p95_ttft, 2),
        p99_ttft_ms=round(p99_ttft, 2),
        mean_worker_decode_tps=round(mean_worker_decode_tps, 2),
        degradation_percent=degradation,
        results=results,
    )
