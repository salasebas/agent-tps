"""Privacy-first local storage for agent-tps benchmark metrics.

Stores only performance telemetry (TPS, TTFT, latencies, tokens, errors).
Zero chat transcripts, prompts, or LLM generated outputs are retained on disk.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from agent_tps.config import get_runs_dir
from agent_tps.core.models import BenchmarkResult, ConcurrencyReport


class BenchmarkStorage:
    """Manages local JSON metric records in the platform application data directory."""

    def __init__(self, runs_dir: Path | None = None):
        self.runs_dir = runs_dir or get_runs_dir()
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def save_run(self, result: BenchmarkResult) -> Path:
        """Saves benchmark performance metrics to disk.

        Strict Privacy Guarantee: Does NOT write raw prompt or chat output.
        """
        timestamp_slug = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp_slug}_{result.provider}_{result.id}.json"
        target_path = self.runs_dir / filename

        # Keep pure numerical and diagnostic telemetry only
        payload: dict[str, Any] = {
            "id": result.id,
            "type": "benchmark",
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "provider": result.provider,
            "model": result.model,
            "status": result.status.value,
            "timeout_type": result.timeout_type.value,
            "error_message": result.error_message,
            "tokens": {
                "input_tokens": result.tokens.input_tokens,
                "output_tokens": result.tokens.output_tokens,
                "reasoning_tokens": result.tokens.reasoning_tokens,
                "cached_read_tokens": result.tokens.cached_read_tokens,
                "cached_write_tokens": result.tokens.cached_write_tokens,
                "generated_tokens": result.tokens.generated_tokens,
                "total_tokens": result.tokens.total_tokens,
                "cache_hit_rate": round(result.tokens.cache_hit_rate, 4),
            },
            "timings": {
                "request_start_ms": result.timings.request_start_ms,
                "ttft_ms": result.timings.ttft_ms,
                "generation_duration_ms": result.timings.generation_duration_ms,
                "total_duration_ms": result.timings.total_duration_ms,
                "itl_p50_ms": result.timings.itl_p50_ms,
                "itl_p90_ms": result.timings.itl_p90_ms,
                "itl_p95_ms": result.timings.itl_p95_ms,
                "itl_p99_ms": result.timings.itl_p99_ms,
                "itl_max_ms": result.timings.itl_max_ms,
                "jitter_ms": result.timings.jitter_ms,
            },
            "tps": {
                "decode_tps": result.tps.decode_tps,
                "e2e_tps": result.tps.e2e_tps,
                "total_throughput_tps": result.tps.total_throughput_tps,
            },
            "metadata": result.metadata,
        }

        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        return target_path

    def save_concurrency_run(
        self,
        report: ConcurrencyReport,
        target: str,
        model: str | None = None,
    ) -> Path:
        """Saves concurrency stress test performance metrics to disk."""
        timestamp_slug = time.strftime("%Y%m%d_%H%M%S")
        run_id = f"stress_{target}_{int(time.time() * 1000)}"
        filename = f"{timestamp_slug}_stress_{target}_{run_id}.json"
        target_path = self.runs_dir / filename

        sample_errors: list[str] = []
        for r in report.results:
            if r.error_message and r.error_message not in sample_errors:
                sample_errors.append(r.error_message)

        status_str = "SUCCESS" if report.successful_requests > 0 else "FAILED"

        payload: dict[str, Any] = {
            "id": run_id,
            "type": "stress",
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "provider": target,
            "model": model or "auto",
            "status": status_str,
            "concurrency_level": report.concurrency_level,
            "total_requests": report.total_requests,
            "successful_requests": report.successful_requests,
            "failed_requests": report.failed_requests,
            "timed_out_requests": report.timed_out_requests,
            "rate_limited_requests": report.rate_limited_requests,
            "wall_clock_duration_s": report.wall_clock_duration_s,
            "aggregate_decode_tps": report.aggregate_decode_tps,
            "aggregate_e2e_tps": report.aggregate_e2e_tps,
            "mean_worker_decode_tps": report.mean_worker_decode_tps,
            "mean_ttft_ms": report.mean_ttft_ms,
            "p50_ttft_ms": report.p50_ttft_ms,
            "p95_ttft_ms": report.p95_ttft_ms,
            "p99_ttft_ms": report.p99_ttft_ms,
            "degradation_percent": report.degradation_percent,
            "sample_errors": sample_errors[:5],
            "tps": {
                "decode_tps": report.aggregate_decode_tps,
                "e2e_tps": report.aggregate_e2e_tps,
            },
            "timings": {
                "ttft_ms": report.mean_ttft_ms,
                "total_duration_ms": round(report.wall_clock_duration_s * 1000.0, 1),
            },
        }

        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        return target_path

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        """Returns the most recent saved runs sorted newest first."""
        files = sorted(self.runs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        runs: list[dict[str, Any]] = []

        for p in files[:limit]:
            try:
                with open(p, encoding="utf-8") as f:
                    data = json.load(f)
                    data["_filepath"] = str(p)
                    data["_filename"] = p.name
                    runs.append(data)
            except Exception:
                continue

        return runs

    def get_run(self, identifier: str) -> dict[str, Any] | None:
        """Finds a run by filename prefix or run ID."""
        for p in self.runs_dir.glob("*.json"):
            if identifier in p.stem or identifier == p.name:
                try:
                    with open(p, encoding="utf-8") as f:
                        data = json.load(f)
                        data["_filepath"] = str(p)
                        data["_filename"] = p.name
                        return data
                except Exception:
                    return None
        return None

    def delete_run(self, identifier: str) -> bool:
        """Deletes a specific benchmark run file."""
        for p in self.runs_dir.glob("*.json"):
            if identifier in p.stem or identifier == p.name:
                try:
                    p.unlink(missing_ok=True)
                    return True
                except Exception:
                    return False
        return False

    def clear_all_runs(self) -> int:
        """Deletes all saved benchmark files and returns the count of deleted files."""
        count = 0
        for p in self.runs_dir.glob("*.json"):
            try:
                p.unlink(missing_ok=True)
                count += 1
            except Exception:
                pass
        return count
