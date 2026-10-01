from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import time
from typing import Any, Callable

from agent_tps_bench.calculator import compute_timing_metrics, compute_tps_metrics
from agent_tps_bench.models import (
    BenchmarkResult,
    BenchmarkStatus,
    TimeoutType,
    TimingMetrics,
    TokenMetrics,
    TPSMetrics,
)
from agent_tps_bench.opencode_reader import OpenCodeDBReader


class OpenCodeRunner:
    """Executes active benchmark prompts through OpenCode CLI with high-resolution streaming metrics."""

    def __init__(
        self,
        binary_path: str = "opencode",
        db_reader: OpenCodeDBReader | None = None,
    ):
        self.binary_path = binary_path
        self.db_reader = db_reader or OpenCodeDBReader()

    async def run_prompt(
        self,
        prompt: str,
        model: str | None = None,
        working_dir: str | Path | None = None,
        ttft_timeout_s: float = 15.0,
        stall_timeout_s: float = 10.0,
        deadline_timeout_s: float = 60.0,
        on_chunk: Callable[[str], None] | None = None,
    ) -> BenchmarkResult:
        """Runs a prompt against OpenCode, monitoring JSON event streams and measuring TPS & timeouts."""
        working_directory = str(working_dir) if working_dir else str(Path.cwd())
        cmd = [
            self.binary_path,
            "run",
            "--format",
            "json",
            "--dir",
            working_directory,
        ]
        if model:
            cmd.extend(["-m", model])
        cmd.append(prompt)

        req_id = f"opencode_{int(time.time() * 1000)}"
        t0 = time.perf_counter()
        request_start_ms = time.time() * 1000.0

        t_first_token: float | None = None
        t_last_token: float | None = None
        inter_token_latencies: list[float] = []
        tokens_captured = TokenMetrics()
        captured_session_id: str | None = None
        text_preview_parts: list[str] = []

        timed_out = False
        timeout_reason = TimeoutType.NONE
        error_message: str | None = None

        max_retries = 3
        for attempt in range(max_retries):
            timed_out = False
            timeout_reason = TimeoutType.NONE
            error_message = None
            t_first_token = None
            t_last_token = None
            inter_token_latencies.clear()
            tokens_captured = TokenMetrics()
            captured_session_id = None
            text_preview_parts.clear()
            received_any_content = False

            t0 = time.perf_counter()
            request_start_ms = time.time() * 1000.0

            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=os.environ.copy(),
                )
            except FileNotFoundError:
                return BenchmarkResult(
                    id=req_id,
                    provider="opencode",
                    model=model or "default",
                    status=BenchmarkStatus.ERROR,
                    timeout_type=TimeoutType.PROCESS_CRASH,
                    error_message=f"Binary not found: {self.binary_path}",
                    timings=TimingMetrics(request_start_ms=request_start_ms),
                )
            except Exception as e:
                return BenchmarkResult(
                    id=req_id,
                    provider="opencode",
                    model=model or "default",
                    status=BenchmarkStatus.ERROR,
                    timeout_type=TimeoutType.UNKNOWN_ERROR,
                    error_message=str(e),
                    timings=TimingMetrics(request_start_ms=request_start_ms),
                )

            assert proc.stdout is not None
            assert proc.stderr is not None
            last_activity_time = t0

            async def read_stream():
                nonlocal t_first_token, t_last_token, last_activity_time, received_any_content
                nonlocal captured_session_id, tokens_captured, text_preview_parts

                while True:
                    line = await proc.stdout.readline()
                    if not line:
                        break
                    now = time.perf_counter()
                    line_str = line.decode("utf-8", errors="replace").strip()
                    if not line_str:
                        continue

                    try:
                        event = json.loads(line_str)
                    except Exception:
                        continue

                    if "sessionID" in event and not captured_session_id:
                        captured_session_id = event["sessionID"]

                    event_type = event.get("type")
                    part = event.get("part", {})
                    part_type = part.get("type")

                    # Detect token/content arrival
                    if event_type in ("text", "reasoning") or part_type in ("text", "reasoning"):
                        chunk_text = part.get("text", "")
                        if chunk_text:
                            text_preview_parts.append(chunk_text)
                            if on_chunk:
                                on_chunk(chunk_text)

                        if not received_any_content:
                            received_any_content = True
                            t_first_token = now
                            last_activity_time = now
                        else:
                            delta_ms = (now - last_activity_time) * 1000.0
                            inter_token_latencies.append(delta_ms)
                            last_activity_time = now

                        t_last_token = now

                    elif event_type == "step_finish" or part_type == "step-finish":
                        token_data = part.get("tokens", {})
                        cache_data = token_data.get("cache", {})
                        tokens_captured = TokenMetrics(
                            input_tokens=token_data.get("input", 0),
                            output_tokens=token_data.get("output", 0),
                            reasoning_tokens=token_data.get("reasoning", 0),
                            cached_read_tokens=cache_data.get("read", 0),
                            cached_write_tokens=cache_data.get("write", 0),
                        )
                        t_last_token = now

            stream_task = asyncio.create_task(read_stream())

            try:
                while not stream_task.done():
                    await asyncio.sleep(0.05)
                    now = time.perf_counter()

                    if (now - t0) > deadline_timeout_s:
                        timed_out = True
                        timeout_reason = TimeoutType.DEADLINE_TIMEOUT
                        error_message = f"Total deadline of {deadline_timeout_s}s exceeded"
                        break

                    if not received_any_content and (now - t0) > ttft_timeout_s:
                        timed_out = True
                        timeout_reason = TimeoutType.TTFT_TIMEOUT
                        error_message = f"Time to First Token exceeded threshold of {ttft_timeout_s}s"
                        break

                    if received_any_content and (now - last_activity_time) > stall_timeout_s:
                        timed_out = True
                        timeout_reason = TimeoutType.STALL_TIMEOUT
                        error_message = f"Token stall detected: no content for {stall_timeout_s}s"
                        break

                if timed_out:
                    stream_task.cancel()
                    try:
                        proc.terminate()
                        await asyncio.wait_for(proc.wait(), timeout=1.5)
                    except Exception:
                        proc.kill()
                else:
                    await stream_task
                    await proc.wait()

            except asyncio.CancelledError:
                proc.kill()
                raise

            t_end = time.perf_counter()
            stderr_bytes = await proc.stderr.read()
            stderr_str = stderr_bytes.decode("utf-8", errors="replace").strip()

            # Check if retryable SQLite lock occurred
            if proc.returncode != 0 and "database is locked" in stderr_str.lower() and attempt < max_retries - 1:
                import random
                await asyncio.sleep(0.2 * (2 ** attempt) + random.uniform(0.05, 0.2))
                continue
            break

        # Handle process exit codes and error classifications
        if proc.returncode != 0 and not timed_out:
            error_lower = stderr_str.lower()
            if "429" in error_lower or "rate limit" in error_lower or "quota" in error_lower:
                status = BenchmarkStatus.RATE_LIMITED
                timeout_reason = TimeoutType.RATE_LIMIT_429
            elif "500" in error_lower or "502" in error_lower or "503" in error_lower or "504" in error_lower:
                status = BenchmarkStatus.ERROR
                timeout_reason = TimeoutType.SERVER_ERROR_5XX
            else:
                status = BenchmarkStatus.ERROR
                timeout_reason = TimeoutType.PROCESS_CRASH
            error_message = stderr_str or f"OpenCode exited with code {proc.returncode}"
        elif timed_out:
            status = BenchmarkStatus.TIMEOUT
        else:
            status = BenchmarkStatus.SUCCESS

        # Calculate high-precision millisecond timestamps
        first_token_ms = (
            request_start_ms + ((t_first_token - t0) * 1000.0)
            if t_first_token is not None
            else None
        )
        completed_ms = request_start_ms + ((t_end - t0) * 1000.0)

        # Cross-verify with OpenCode SQLite DB if session ID was found
        db_detail = None
        if captured_session_id and self.db_reader.is_available():
            try:
                db_detail = self.db_reader.get_session_by_id(captured_session_id)
                if db_detail and db_detail.tokens.total_tokens > 0:
                    tokens_captured = db_detail.tokens
            except Exception:
                pass

        # If step_finish didn't supply tokens and DB didn't either, estimate
        if tokens_captured.output_tokens == 0 and text_preview_parts:
            full_text = "".join(text_preview_parts)
            # Standard heuristic: 4 chars per token
            estimated_tokens = max(1, len(full_text) // 4)
            tokens_captured.output_tokens = estimated_tokens

        timings = compute_timing_metrics(
            request_start_ms=request_start_ms,
            first_token_ms=first_token_ms,
            completed_ms=completed_ms,
            inter_token_latencies_ms=inter_token_latencies,
        )

        tps = compute_tps_metrics(tokens_captured, timings)

        return BenchmarkResult(
            id=captured_session_id or req_id,
            provider="opencode",
            model=model or (db_detail.model if db_detail else "default"),
            status=status,
            timeout_type=timeout_reason,
            tokens=tokens_captured,
            timings=timings,
            tps=tps,
            error_message=error_message,
            raw_response_preview="".join(text_preview_parts)[:200],
            metadata={
                "session_id": captured_session_id,
                "cost_usd": db_detail.cost if db_detail else 0.0,
                "agent": db_detail.agent if db_detail else "build",
            },
        )
