from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any
from agent_tps_bench.models import BenchmarkResult


DEFAULT_STORAGE_DIR = Path.home() / ".agent-tps-bench" / "runs"


class BenchmarkStorage:
    """Persists and queries benchmark runs with full prompts, outputs, and metrics."""

    def __init__(self, storage_dir: Path | str | None = None):
        self.storage_dir = Path(storage_dir) if storage_dir else DEFAULT_STORAGE_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save_run(
        self,
        result: BenchmarkResult,
        prompt: str,
        full_output: str = "",
        extra_meta: dict[str, Any] | None = None,
    ) -> Path:
        """Saves a benchmark execution to a timestamped JSON file."""
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        sanitized_provider = result.provider.replace("/", "_").replace(":", "_")
        sanitized_model = result.model.replace("/", "_").replace(":", "_")
        sanitized_id = result.id.replace("/", "_").replace(":", "_")
        filename = f"{timestamp_str}_{sanitized_provider}_{sanitized_model}_{sanitized_id}.json"
        target_path = self.storage_dir / filename

        payload = {
            "timestamp": time.time(),
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "prompt": prompt,
            "full_output": full_output or result.raw_response_preview,
            "result": result.model_dump(),
            "extra_meta": extra_meta or {},
        }

        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        return target_path

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        """Returns the most recent runs summary."""
        files = sorted(self.storage_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        summaries: list[dict[str, Any]] = []

        for p in files[:limit]:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    res = data.get("result", {})
                    tps = res.get("tps", {})
                    timings = res.get("timings", {})
                    tok = res.get("tokens", {})
                    summaries.append({
                        "file_path": str(p),
                        "id": res.get("id", p.stem),
                        "timestamp_iso": data.get("timestamp_iso", ""),
                        "provider": res.get("provider", "unknown"),
                        "model": res.get("model", "unknown"),
                        "status": res.get("status", "unknown"),
                        "decode_tps": tps.get("decode_tps", 0.0),
                        "e2e_tps": tps.get("e2e_tps", 0.0),
                        "ttft_ms": timings.get("ttft_ms"),
                        "tokens_generated": tok.get("output_tokens", 0) + tok.get("reasoning_tokens", 0),
                        "prompt": data.get("prompt", "")[:60],
                    })
            except Exception:
                continue

        return summaries

    def load_run(self, file_path_or_id: str) -> dict[str, Any] | None:
        """Loads a run by file path or ID prefix."""
        p = Path(file_path_or_id)
        if p.exists() and p.is_file():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)

        # Search candidates
        for candidate in self.storage_dir.glob("*.json"):
            if file_path_or_id in candidate.name:
                with open(candidate, "r", encoding="utf-8") as f:
                    return json.load(f)
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data.get("result", {}).get("id") == file_path_or_id:
                        return data
            except Exception:
                continue
        return None
