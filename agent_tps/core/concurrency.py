from __future__ import annotations

import asyncio
from collections.abc import Callable
import time
from typing import Any

from agent_tps.core.calculator import aggregate_concurrency_results
from agent_tps.core.models import BenchmarkResult, ConcurrencyReport


class ConcurrencyRunner:
    """Orchestrates concurrent subagents or stream workers to measure throughput scaling, degradation, and rate limits."""

    def __init__(self, runner: Any):
        self.runner = runner

    async def run_batch(
        self,
        concurrency: int,
        total_requests: int,
        prompt: str | list[str],
        model: str | None = None,
        baseline_single_worker_tps: float | None = None,
        on_complete: Callable[[BenchmarkResult, int, int], None] | None = None,
        **runner_kwargs: Any,
    ) -> ConcurrencyReport:
        """Executes a pool of concurrent requests, collecting timings and detecting degradation."""
        semaphore = asyncio.Semaphore(concurrency)
        prompts = [prompt] * total_requests if isinstance(prompt, str) else prompt
        if len(prompts) < total_requests:
            prompts = (prompts * ((total_requests // len(prompts)) + 1))[:total_requests]

        completed_count = 0
        results: list[BenchmarkResult] = []
        lock = asyncio.Lock()

        async def worker(idx: int, p: str):
            nonlocal completed_count
            async with semaphore:
                try:
                    if hasattr(self.runner, "run_prompt"):
                        res = await self.runner.run_prompt(
                            prompt=p,
                            model=model,
                            **runner_kwargs,
                        )
                    else:
                        res = await self.runner.run_stream(
                            model=model or "default",
                            prompt=p,
                            **runner_kwargs,
                        )
                except Exception as exc:
                    from agent_tps.core.models import BenchmarkStatus, TimeoutType, TimingMetrics

                    res = BenchmarkResult(
                        id=f"err_{idx}_{int(time.time() * 1000)}",
                        provider="error",
                        model=model or "unknown",
                        status=BenchmarkStatus.ERROR,
                        timeout_type=TimeoutType.UNKNOWN_ERROR,
                        error_message=str(exc),
                        timings=TimingMetrics(request_start_ms=time.time() * 1000.0),
                    )

                async with lock:
                    completed_count += 1
                    results.append(res)
                    if on_complete:
                        on_complete(res, completed_count, total_requests)
                return res

        t_start = time.perf_counter()
        tasks = [asyncio.create_task(worker(i, prompts[i])) for i in range(total_requests)]
        await asyncio.gather(*tasks, return_exceptions=True)
        t_end = time.perf_counter()

        wall_clock = t_end - t_start

        return aggregate_concurrency_results(
            concurrency_level=concurrency,
            results=results,
            wall_clock_duration_s=wall_clock,
            baseline_decode_tps=baseline_single_worker_tps,
        )

    async def run_sweep(
        self,
        concurrency_levels: list[int],
        requests_per_level: int,
        prompt: str,
        model: str | None = None,
        on_level_complete: Callable[[ConcurrencyReport], None] | None = None,
        **runner_kwargs: Any,
    ) -> list[ConcurrencyReport]:
        """Runs a progressive sweep across multiple concurrency levels (e.g. 1, 2, 4, 8) to trace scalability."""
        reports: list[ConcurrencyReport] = []
        baseline_tps: float | None = None

        for idx, c in enumerate(concurrency_levels):
            report = await self.run_batch(
                concurrency=c,
                total_requests=requests_per_level,
                prompt=prompt,
                model=model,
                baseline_single_worker_tps=baseline_tps,
                **runner_kwargs,
            )
            if idx == 0:
                baseline_tps = report.mean_worker_decode_tps

            reports.append(report)
            if on_level_complete:
                on_level_complete(report)

        return reports
