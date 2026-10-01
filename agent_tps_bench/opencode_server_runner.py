from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import time
from typing import Any, Callable
import httpx

from agent_tps_bench.calculator import compute_timing_metrics, compute_tps_metrics
from agent_tps_bench.models import (
    BenchmarkResult,
    BenchmarkStatus,
    TimeoutType,
    TimingMetrics,
    TokenMetrics,
    TPSMetrics,
)


class OpenCodeServerRunner:
    """Interacts with OpenCode through its HTTP server (opencode serve), matching T3 Code's architecture.

    Allows concurrent subagents to run without SQLite database-locking conflicts.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:4096",
        auto_spawn: bool = True,
        port: int = 4096,
        binary_path: str = "opencode",
    ):
        self.base_url = base_url.rstrip("/")
        self.auto_spawn = auto_spawn
        self.port = port
        self.binary_path = binary_path
        self._server_proc: asyncio.subprocess.Process | None = None

    async def is_healthy(self) -> bool:
        """Checks if the OpenCode server is responding to health checks."""
        try:
            async with httpx.AsyncClient(timeout=1.5) as client:
                res = await client.get(f"{self.base_url}/global/health")
                if res.status_code == 200:
                    data = res.json()
                    return bool(data.get("healthy"))
        except Exception:
            return False
        return False

    async def ensure_server(self, timeout_s: float = 10.0) -> None:
        """Ensures the OpenCode server is running; spawns it if auto_spawn is enabled."""
        if await self.is_healthy():
            return

        if not self.auto_spawn:
            raise RuntimeError(f"OpenCode server not running at {self.base_url}")

        cmd = [self.binary_path, "serve", "--port", str(self.port)]
        self._server_proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ.copy(),
        )

        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout_s:
            if await self.is_healthy():
                return
            await asyncio.sleep(0.3)

        raise TimeoutError(f"Failed to start OpenCode server within {timeout_s}s")

    async def stop_server(self) -> None:
        """Stops the spawned server process if one was started."""
        if self._server_proc:
            try:
                self._server_proc.terminate()
                await asyncio.wait_for(self._server_proc.wait(), timeout=2.0)
            except Exception:
                try:
                    self._server_proc.kill()
                except Exception:
                    pass
            self._server_proc = None

    async def create_session(self, title: str = "TPS Benchmark") -> str:
        """Creates a new session on the OpenCode server."""
        await self.ensure_server()
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.post(
                f"{self.base_url}/session",
                json={"title": title, "permission": [{"permission": "*", "pattern": "*", "action": "deny"}]},
            )
            res.raise_for_status()
            data = res.json()
            return data["id"]

    async def run_prompt(
        self,
        prompt: str,
        model: str | None = None,
        ttft_timeout_s: float = 15.0,
        stall_timeout_s: float = 10.0,
        deadline_timeout_s: float = 60.0,
        on_chunk: Callable[[str], None] | None = None,
    ) -> BenchmarkResult:
        """Executes a prompt against OpenCode Server and measures latency, TTFT, and TPS."""
        await self.ensure_server()
        session_id = await self.create_session(f"Bench {int(time.time())}")
        req_id = f"opencode_srv_{int(time.time() * 1000)}"

        t0 = time.perf_counter()
        request_start_ms = time.time() * 1000.0

        t_connected: float | None = None
        t_first_token: float | None = None
        t_last_token: float | None = None
        inter_token_latencies: list[float] = []
        tokens_captured = TokenMetrics()
        text_preview_parts: list[str] = []

        timed_out = False
        timeout_reason = TimeoutType.NONE
        status = BenchmarkStatus.SUCCESS
        error_message: str | None = None

        # Format model payload if supplied (e.g. "opencode/longcat-2.5-preview-free")
        parsed_model = None
        if model:
            if "/" in model:
                prov_id, mod_id = model.split("/", 1)
                parsed_model = {"providerID": prov_id, "modelID": mod_id}
            else:
                parsed_model = {"providerID": "opencode", "modelID": model}

        prompt_payload = {
            "parts": [{"type": "text", "text": prompt}],
        }
        if parsed_model:
            prompt_payload["model"] = parsed_model

        try:
            # We call /session/{session_id}/prompt which returns when generation completes
            timeout_config = httpx.Timeout(
                connect=ttft_timeout_s,
                read=deadline_timeout_s,
                write=10.0,
                pool=10.0,
            )
            async with httpx.AsyncClient(timeout=timeout_config) as client:
                t_connected = time.perf_counter()
                post_task = asyncio.create_task(
                    client.post(f"{self.base_url}/session/{session_id}/message", json=prompt_payload)
                )

                # Wait for completion while checking deadline
                now = time.perf_counter()
                while not post_task.done():
                    await asyncio.sleep(0.05)
                    now = time.perf_counter()
                    if (now - t0) > deadline_timeout_s:
                        timed_out = True
                        timeout_reason = TimeoutType.DEADLINE_TIMEOUT
                        error_message = f"Deadline of {deadline_timeout_s}s exceeded"
                        post_task.cancel()
                        break

                if not timed_out:
                    resp = await post_task
                    t_end = time.perf_counter()

                    if resp.status_code == 429:
                        status = BenchmarkStatus.RATE_LIMITED
                        timeout_reason = TimeoutType.RATE_LIMIT_429
                        error_message = f"Rate limited (429): {resp.text[:200]}"
                    elif resp.status_code >= 500:
                        status = BenchmarkStatus.ERROR
                        timeout_reason = TimeoutType.SERVER_ERROR_5XX
                        error_message = f"Server Error ({resp.status_code}): {resp.text[:200]}"
                    elif resp.status_code != 200:
                        status = BenchmarkStatus.ERROR
                        timeout_reason = TimeoutType.CLIENT_ERROR_4XX
                        error_message = f"HTTP Error ({resp.status_code}): {resp.text[:200]}"
                    else:
                        resp_data = resp.json()
                        info = resp_data.get("info", {})
                        
                        # Detect OpenCode provider errors embedded in HTTP 200 responses
                        opencode_err = info.get("error")
                        if opencode_err:
                            err_data = opencode_err.get("data", {}) if isinstance(opencode_err, dict) else {}
                            err_msg = err_data.get("message") or str(opencode_err)
                            status = BenchmarkStatus.ERROR
                            status_code = err_data.get("statusCode", 400)
                            if status_code == 429:
                                status = BenchmarkStatus.RATE_LIMITED
                                timeout_reason = TimeoutType.RATE_LIMIT_429
                            elif status_code >= 500:
                                timeout_reason = TimeoutType.SERVER_ERROR_5XX
                            else:
                                timeout_reason = TimeoutType.CLIENT_ERROR_4XX
                            error_message = f"Provider Error ({status_code}): {err_msg}"

                        tok_info = info.get("tokens", {})
                        cache_info = tok_info.get("cache", {})

                        tokens_captured = TokenMetrics(
                            input_tokens=tok_info.get("input", 0),
                            output_tokens=tok_info.get("output", 0),
                            reasoning_tokens=tok_info.get("reasoning", 0),
                            cached_read_tokens=cache_info.get("read", 0),
                            cached_write_tokens=cache_info.get("write", 0),
                        )

                        # Check exact parts and timing from OpenCode response
                        parts = resp_data.get("parts", [])
                        part_dur_accum_ms = 0.0
                        first_part_start_ms: float | None = None
                        last_part_end_ms: float | None = None

                        for p in parts:
                            p_type = p.get("type")
                            if p_type in ("text", "reasoning"):
                                p_text = p.get("text", "")
                                if p_text:
                                    text_preview_parts.append(p_text)
                                    if on_chunk:
                                        on_chunk(p_text)
                                p_time = p.get("time") or {}
                                p_start = p_time.get("start")
                                p_end = p_time.get("end")
                                if p_start and p_end:
                                    if first_part_start_ms is None or p_start < first_part_start_ms:
                                        first_part_start_ms = float(p_start)
                                    if last_part_end_ms is None or p_end > last_part_end_ms:
                                        last_part_end_ms = float(p_end)
                                    part_dur_accum_ms += max(0.0, float(p_end - p_start))

                        resp_time = info.get("time", {})
                        p_created = resp_time.get("created")
                        p_completed = resp_time.get("completed")

                        if first_part_start_ms and p_created:
                            ttft_sec = max(0.001, (first_part_start_ms - p_created) / 1000.0)
                            t_first_token = t0 + ttft_sec
                        elif p_created and p_completed:
                            t_gen_s = (p_completed - p_created) / 1000.0
                            t_first_token = t0 + max(0.001, ((t_end - t0) - t_gen_s))
                        else:
                            t_first_token = t0 + ((t_end - t0) * 0.7)

                        t_last_token = t_end

        except httpx.ConnectTimeout:
            timed_out = True
            timeout_reason = TimeoutType.CONNECT_TIMEOUT
            error_message = f"Connect timeout exceeded ({ttft_timeout_s}s)"
        except Exception as e:
            if not timed_out:
                status = BenchmarkStatus.ERROR
                timeout_reason = TimeoutType.UNKNOWN_ERROR
                error_message = str(e)

        t_end = time.perf_counter()
        if timed_out:
            status = BenchmarkStatus.TIMEOUT

        first_token_ms = (
            request_start_ms + ((t_first_token - t0) * 1000.0)
            if t_first_token is not None
            else None
        )
        completed_ms = request_start_ms + ((t_end - t0) * 1000.0)
        connection_ms = (
            request_start_ms + ((t_connected - t0) * 1000.0)
            if t_connected is not None
            else None
        )

        timings = compute_timing_metrics(
            request_start_ms=request_start_ms,
            first_token_ms=first_token_ms,
            completed_ms=completed_ms,
            inter_token_latencies_ms=inter_token_latencies,
            connection_ms=connection_ms,
        )

        # Fallback estimation if zero tokens reported
        if tokens_captured.output_tokens == 0 and text_preview_parts:
            tokens_captured.output_tokens = max(1, len("".join(text_preview_parts)) // 4)

        tps = compute_tps_metrics(tokens_captured, timings)

        return BenchmarkResult(
            id=session_id or req_id,
            provider="opencode-server",
            model=model or "default",
            status=status,
            timeout_type=timeout_reason,
            tokens=tokens_captured,
            timings=timings,
            tps=tps,
            error_message=error_message,
            raw_response_preview="".join(text_preview_parts)[:200],
            metadata={"session_id": session_id, "mode": "http-server"},
        )
