import asyncio

import pytest

from tokpulse.core.concurrency import ConcurrencyRunner
from tokpulse.core.models import (
    BenchmarkResult,
    BenchmarkStatus,
    TimeoutType,
    TimingMetrics,
    TokenMetrics,
    TPSMetrics,
)


class MockRunner:
    def __init__(self, delay: float = 0.05, tokens: int = 20, rate_limit_after: int = 999):
        self.delay = delay
        self.tokens = tokens
        self.rate_limit_after = rate_limit_after
        self.count = 0
        self.lock = asyncio.Lock()

    async def run_stream(self, model: str, prompt: str, **kwargs) -> BenchmarkResult:
        async with self.lock:
            self.count += 1
            current = self.count

        if current > self.rate_limit_after:
            return BenchmarkResult(
                id=f"req_{current}",
                provider="mock",
                model=model,
                status=BenchmarkStatus.RATE_LIMITED,
                timeout_type=TimeoutType.RATE_LIMIT_429,
                retry_after_s=2.0,
                timings=TimingMetrics(request_start_ms=0, completed_ms=50),
            )

        await asyncio.sleep(self.delay)
        return BenchmarkResult(
            id=f"req_{current}",
            provider="mock",
            model=model,
            status=BenchmarkStatus.SUCCESS,
            tokens=TokenMetrics(output_tokens=self.tokens),
            timings=TimingMetrics(
                request_start_ms=0,
                first_token_ms=10,
                completed_ms=int(self.delay * 1000),
                generation_duration_ms=int(self.delay * 1000) - 10,
                total_duration_ms=int(self.delay * 1000),
                ttft_ms=10.0,
            ),
            tps=TPSMetrics(
                decode_tps=self.tokens / (self.delay - 0.01),
                e2e_tps=self.tokens / self.delay,
            ),
        )


@pytest.mark.asyncio
async def test_concurrency_batch_execution():
    mock_runner = MockRunner(delay=0.04, tokens=40)
    orchestrator = ConcurrencyRunner(runner=mock_runner)

    completed_events = []
    report = await orchestrator.run_batch(
        concurrency=3,
        total_requests=6,
        prompt="test prompt",
        on_complete=lambda res, comp, tot: completed_events.append(comp),
    )

    assert report.total_requests == 6
    assert report.successful_requests == 6
    assert len(completed_events) == 6
    assert report.aggregate_decode_tps > 0.0
    assert report.mean_worker_decode_tps > 0.0


@pytest.mark.asyncio
async def test_concurrency_rate_limit_detection():
    # Only allow 2 requests before rate limiting
    mock_runner = MockRunner(delay=0.02, tokens=10, rate_limit_after=2)
    orchestrator = ConcurrencyRunner(runner=mock_runner)

    report = await orchestrator.run_batch(
        concurrency=2,
        total_requests=4,
        prompt="test",
    )

    assert report.total_requests == 4
    assert report.successful_requests == 2
    assert report.rate_limited_requests == 2
