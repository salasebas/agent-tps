from __future__ import annotations

import asyncio
from collections.abc import Callable
import os
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


class GrokRunner(BaseAgentRunner):
    """Executes benchmark prompts through xAI's Grok CLI (grok or grok-build)."""

    def __init__(self, binary_path: str | None = None, auto_cleanup: bool = True):
        self.binary_path = binary_path or self._detect_binary()
        self.auto_cleanup = auto_cleanup
        self._temp_dirs: list[str] = []

    def _detect_binary(self) -> str:
        candidates = [
            "grok-build",
            "grok",
            str(os.path.expanduser("~/.grok/bin/grok")),
            str(os.path.expanduser("~/.local/bin/grok")),
            "/usr/local/bin/grok",
            "/opt/homebrew/bin/grok",
        ]
        for c in candidates:
            if shutil.which(c) or (os.path.isfile(c) and os.access(c, os.X_OK)):
                return c
        return "grok"

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
        deadline_timeout_s: float = 60.0,
        on_chunk: Callable[[str], None] | None = None,
    ) -> BenchmarkResult:
        req_id = f"grok_{int(time.time() * 1000)}"
        request_start_ms = time.time() * 1000.0

        resolved_bin = shutil.which(self.binary_path) or (
            self.binary_path if os.path.isfile(self.binary_path) else None
        )
        if not resolved_bin:
            return BenchmarkResult(
                id=req_id,
                provider="grok",
                model=model or "grok-build",
                status=BenchmarkStatus.ERROR,
                timeout_type=TimeoutType.PROCESS_CRASH,
                error_message=f"Grok binary '{self.binary_path}' not found in PATH.",
                timings=compute_timing_metrics(request_start_ms=request_start_ms),
            )

        temp_dir = tempfile.mkdtemp(prefix="agent_tps_grok_")
        self._temp_dirs.append(temp_dir)

        cmd = [resolved_bin, "-p", prompt, "--output-format", "plain"]
        if model:
            cmd.extend(["-m", model])

        t0 = time.perf_counter()
        tokens_captured = TokenMetrics()

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=temp_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=os.environ.copy(),
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=deadline_timeout_s,
            )
            completed_ms = time.time() * 1000.0
            t_first_token = request_start_ms + ((time.perf_counter() - t0) * 500.0)

            stdout_str = stdout_bytes.decode("utf-8", errors="replace").strip()
            stderr_str = stderr_bytes.decode("utf-8", errors="replace").strip()

            if proc.returncode != 0:
                if self.auto_cleanup:
                    await self.cleanup_session()
                return BenchmarkResult(
                    id=req_id,
                    provider="grok",
                    model=model or "grok-build",
                    status=BenchmarkStatus.ERROR,
                    timeout_type=TimeoutType.PROCESS_CRASH,
                    error_message=stderr_str or f"Process exited with code {proc.returncode}",
                    timings=compute_timing_metrics(
                        request_start_ms=request_start_ms, completed_ms=completed_ms
                    ),
                )

            if on_chunk and stdout_str:
                on_chunk(stdout_str)

            tokens_captured.output_tokens = max(1, len(stdout_str.split()))

            timings = compute_timing_metrics(
                request_start_ms=request_start_ms,
                first_token_ms=t_first_token,
                completed_ms=completed_ms,
            )
            tps = compute_tps_metrics(tokens_captured, timings)

            if self.auto_cleanup:
                await self.cleanup_session()

            return BenchmarkResult(
                id=req_id,
                provider="grok",
                model=model or "grok-build",
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
                provider="grok",
                model=model or "grok-build",
                status=BenchmarkStatus.TIMEOUT,
                timeout_type=TimeoutType.DEADLINE_TIMEOUT,
                error_message=f"Execution exceeded deadline timeout ({deadline_timeout_s}s)",
                timings=compute_timing_metrics(
                    request_start_ms=request_start_ms,
                    completed_ms=request_start_ms + (deadline_timeout_s * 1000.0),
                ),
            )
