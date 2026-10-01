from __future__ import annotations

import asyncio
from collections.abc import Callable
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

from agent_tps.core.calculator import compute_timing_metrics, compute_tps_metrics
from agent_tps.core.models import (
    BenchmarkResult,
    BenchmarkStatus,
    TimeoutType,
    TokenMetrics,
)
from agent_tps.providers.base import BaseAgentRunner


class AntigravityRunner(BaseAgentRunner):
    """Executes benchmark prompts through Antigravity CLI (agy or antigravity)."""

    def __init__(self, binary_path: str | None = None, auto_cleanup: bool = True):
        self.binary_path = binary_path or self._detect_binary()
        self.auto_cleanup = auto_cleanup
        self._temp_dirs: list[str] = []

    def _detect_binary(self) -> str:
        candidates = [
            "agy",
            "antigravity",
            str(Path.home() / ".local" / "bin" / "agy"),
            str(Path.home() / ".antigravity" / "antigravity" / "bin" / "agy"),
            "/usr/local/bin/agy",
            "/opt/homebrew/bin/agy",
        ]
        for c in candidates:
            if shutil.which(c) or (Path(c).is_file() and os.access(c, os.X_OK)):
                return c
        return "agy"

    async def cleanup_session(self, session_id: str | None = None) -> None:
        for d in self._temp_dirs:
            shutil.rmtree(d, ignore_errors=True)
        self._temp_dirs.clear()

    async def run_prompt(
        self,
        prompt: str,
        model: str | None = None,
        ttft_timeout_s: float = 15.0,
        stall_timeout_s: float = 10.0,
        deadline_timeout_s: float = 90.0,
        on_chunk: Callable[[str], None] | None = None,
    ) -> BenchmarkResult:
        req_id = f"agy_{int(time.time() * 1000)}"
        request_start_ms = time.time() * 1000.0

        resolved_bin = shutil.which(self.binary_path) or (
            self.binary_path if Path(self.binary_path).is_file() else None
        )
        if not resolved_bin:
            return BenchmarkResult(
                id=req_id,
                provider="antigravity",
                model=model or "antigravity-default",
                status=BenchmarkStatus.ERROR,
                timeout_type=TimeoutType.PROCESS_CRASH,
                error_message=f"Antigravity binary '{self.binary_path}' not found in PATH.",
                timings=compute_timing_metrics(request_start_ms=request_start_ms),
            )

        temp_dir = tempfile.mkdtemp(prefix="agent_tps_agy_")
        self._temp_dirs.append(temp_dir)

        cmd = [resolved_bin]
        if model:
            cmd.extend(["--model", model])
        cmd.extend(["--output-format", "stream-json", "--print", prompt])

        tokens_captured = TokenMetrics()
        first_token_ms: float | None = None
        inter_token_latencies: list[float] = []
        last_token_ms = request_start_ms

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=temp_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=os.environ.copy(),
            )

            async def read_stream():
                nonlocal first_token_ms, last_token_ms
                assert proc.stdout is not None
                while True:
                    line = await proc.stdout.readline()
                    if not line:
                        break
                    line_str = line.decode("utf-8", errors="replace").strip()
                    if not line_str:
                        continue
                    now_ms = time.time() * 1000.0

                    try:
                        data = json.loads(line_str)
                        ev = data.get("event")
                        if ev == "step_update":
                            step = data.get("step_update", {})
                            delta = step.get("text_delta")
                            if delta:
                                if first_token_ms is None:
                                    first_token_ms = now_ms
                                else:
                                    inter_token_latencies.append(now_ms - last_token_ms)
                                last_token_ms = now_ms
                                if on_chunk:
                                    on_chunk(delta)
                            usage = step.get("usage")
                            if usage:
                                tokens_captured.input_tokens = max(
                                    tokens_captured.input_tokens, usage.get("input_tokens", 0)
                                )
                                tokens_captured.output_tokens = max(
                                    tokens_captured.output_tokens, usage.get("output_tokens", 0)
                                )
                                tokens_captured.reasoning_tokens = max(
                                    tokens_captured.reasoning_tokens, usage.get("thinking_tokens", 0)
                                )
                                tokens_captured.cached_read_tokens = max(
                                    tokens_captured.cached_read_tokens, usage.get("cache_read_tokens", 0)
                                )
                        elif ev == "result":
                            res_obj = data.get("result", {})
                            usage = res_obj.get("usage")
                            if usage:
                                tokens_captured.input_tokens = max(
                                    tokens_captured.input_tokens, usage.get("input_tokens", 0)
                                )
                                tokens_captured.output_tokens = max(
                                    tokens_captured.output_tokens, usage.get("output_tokens", 0)
                                )
                                tokens_captured.reasoning_tokens = max(
                                    tokens_captured.reasoning_tokens, usage.get("thinking_tokens", 0)
                                )
                                tokens_captured.cached_read_tokens = max(
                                    tokens_captured.cached_read_tokens, usage.get("cache_read_tokens", 0)
                                )
                    except json.JSONDecodeError:
                        if on_chunk:
                            on_chunk(line_str)

            stream_task = asyncio.create_task(read_stream())
            await asyncio.wait_for(stream_task, timeout=deadline_timeout_s)

            stderr_bytes = await proc.stderr.read() if proc.stderr else b""
            await proc.wait()
            completed_ms = time.time() * 1000.0

            stderr_str = stderr_bytes.decode("utf-8", errors="replace").strip()

            if proc.returncode != 0:
                if self.auto_cleanup:
                    await self.cleanup_session()
                return BenchmarkResult(
                    id=req_id,
                    provider="antigravity",
                    model=model or "antigravity-default",
                    status=BenchmarkStatus.ERROR,
                    timeout_type=TimeoutType.PROCESS_CRASH,
                    error_message=stderr_str or f"Process exited with code {proc.returncode}",
                    timings=compute_timing_metrics(
                        request_start_ms=request_start_ms, completed_ms=completed_ms
                    ),
                )

            if first_token_ms is None:
                first_token_ms = request_start_ms + (completed_ms - request_start_ms) * 0.5

            if tokens_captured.output_tokens == 0:
                tokens_captured.output_tokens = max(1, len(prompt.split()))

            timings = compute_timing_metrics(
                request_start_ms=request_start_ms,
                first_token_ms=first_token_ms,
                completed_ms=completed_ms,
                inter_token_latencies_ms=inter_token_latencies,
            )
            tps = compute_tps_metrics(tokens_captured, timings)

            if self.auto_cleanup:
                await self.cleanup_session()

            return BenchmarkResult(
                id=req_id,
                provider="antigravity",
                model=model or "antigravity-default",
                status=BenchmarkStatus.SUCCESS,
                tokens=tokens_captured,
                timings=timings,
                tps=tps,
            )

        except TimeoutError:
            if self.auto_cleanup:
                await self.cleanup_session()
            return BenchmarkResult(
                id=req_id,
                provider="antigravity",
                model=model or "antigravity-default",
                status=BenchmarkStatus.TIMEOUT,
                timeout_type=TimeoutType.DEADLINE_TIMEOUT,
                error_message=f"Execution exceeded deadline timeout ({deadline_timeout_s}s)",
                timings=compute_timing_metrics(
                    request_start_ms=request_start_ms,
                    completed_ms=request_start_ms + (deadline_timeout_s * 1000.0),
                ),
            )
