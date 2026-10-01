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


class CursorRunner(BaseAgentRunner):
    """Executes benchmark prompts through Cursor CLI (cursor-agent)."""

    def __init__(self, binary_path: str | None = None, auto_cleanup: bool = True):
        self.binary_path = binary_path or self._detect_binary()
        self.auto_cleanup = auto_cleanup
        self._temp_dirs: list[str] = []

    def _detect_binary(self) -> str:
        candidates = [
            str(Path.home() / ".local" / "bin" / "cursor-agent"),
            "cursor-agent",
            "/usr/local/bin/cursor-agent",
            "/opt/homebrew/bin/cursor-agent",
            "cursor",
        ]
        for c in candidates:
            if shutil.which(c) or (Path(c).is_file() and os.access(c, os.X_OK)):
                return c
        return "cursor-agent"

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
        req_id = f"cursor_{int(time.time() * 1000)}"
        request_start_ms = time.time() * 1000.0

        resolved_bin = shutil.which(self.binary_path) or (
            self.binary_path if Path(self.binary_path).is_file() else None
        )
        if not resolved_bin:
            return BenchmarkResult(
                id=req_id,
                provider="cursor",
                model=model or "auto",
                status=BenchmarkStatus.ERROR,
                timeout_type=TimeoutType.PROCESS_CRASH,
                error_message=f"Cursor binary '{self.binary_path}' not found in PATH.",
                timings=compute_timing_metrics(request_start_ms=request_start_ms),
            )

        temp_dir = tempfile.mkdtemp(prefix="agent_tps_cursor_")
        self._temp_dirs.append(temp_dir)

        cmd = [resolved_bin, "-p", "--trust", "--output-format", "stream-json"]
        if model and model != "auto":
            cmd.extend(["--model", model])
        cmd.append(prompt)

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
                        msg_type = data.get("type")
                        if msg_type == "thinking" and data.get("subtype") == "delta":
                            delta = data.get("text")
                            if delta:
                                if first_token_ms is None:
                                    first_token_ms = now_ms
                                else:
                                    inter_token_latencies.append(now_ms - last_token_ms)
                                last_token_ms = now_ms
                                if on_chunk:
                                    on_chunk(delta)
                        elif msg_type == "assistant":
                            msg = data.get("message", {})
                            content_list = msg.get("content", [])
                            for c_item in content_list:
                                if c_item.get("type") == "text":
                                    txt = c_item.get("text", "")
                                    if first_token_ms is None:
                                        first_token_ms = now_ms
                                    if on_chunk:
                                        on_chunk(txt)
                        elif msg_type == "result":
                            usage = data.get("usage", {})
                            tokens_captured.input_tokens = usage.get("inputTokens", 0)
                            tokens_captured.output_tokens = usage.get("outputTokens", 0)
                            tokens_captured.cached_read_tokens = usage.get("cacheReadTokens", 0)
                            tokens_captured.cached_write_tokens = usage.get("cacheWriteTokens", 0)
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
                    provider="cursor",
                    model=model or "auto",
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
                provider="cursor",
                model=model or "auto",
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
                provider="cursor",
                model=model or "auto",
                status=BenchmarkStatus.TIMEOUT,
                timeout_type=TimeoutType.DEADLINE_TIMEOUT,
                error_message=f"Execution exceeded deadline timeout ({deadline_timeout_s}s)",
                timings=compute_timing_metrics(
                    request_start_ms=request_start_ms,
                    completed_ms=request_start_ms + (deadline_timeout_s * 1000.0),
                ),
            )
