from __future__ import annotations

import asyncio
from collections.abc import Callable
import contextlib
import os
import time

import httpx

from agent_tps.core.calculator import compute_timing_metrics, compute_tps_metrics
from agent_tps.core.models import (
    BenchmarkResult,
    BenchmarkStatus,
    TimeoutType,
    TokenMetrics,
)
from agent_tps.providers.base import BaseAgentRunner
from agent_tps.providers.opencode import OpenCodeDBReader


class OpenCodeServerRunner(BaseAgentRunner):
    """Interacts with OpenCode through its HTTP server (opencode serve), matching T3 Code's architecture.

    Allows concurrent subagents to run without SQLite database-locking conflicts.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:4096",
        auto_spawn: bool = True,
        port: int = 4096,
        binary_path: str = "opencode",
        auto_cleanup: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.auto_spawn = auto_spawn
        self.port = port
        self.binary_path = binary_path
        self.auto_cleanup = auto_cleanup
        self._server_proc: asyncio.subprocess.Process | None = None
        self._db_reader = OpenCodeDBReader()

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
                with contextlib.suppress(Exception):
                    self._server_proc.kill()
            self._server_proc = None

    async def create_session(self, title: str = "TokPulse Benchmark") -> str:
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

    async def cleanup_session(self, session_id: str | None = None) -> None:
        """Deletes the benchmark session from the server and database for absolute privacy."""
        if not session_id:
            return

        # 1. Attempt HTTP DELETE
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                await client.delete(f"{self.base_url}/session/{session_id}")
        except Exception:
            pass

        # 2. SQLite direct purge fallback
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._db_reader.delete_session, session_id)
        except Exception:
            pass

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
        session_id = await self.create_session(f"TokPulse_{int(time.time())}")
        req_id = f"opencode_srv_{int(time.time() * 1000)}"

        t0 = time.perf_counter()
        request_start_ms = time.time() * 1000.0

        t_connected: float | None = None
        t_first_token: float | None = None
        inter_token_latencies: list[float] = []
        tokens_captured = TokenMetrics()

        timed_out = False
        timeout_reason = TimeoutType.NONE
        status = BenchmarkStatus.SUCCESS
        error_message: str | None = None

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

                        opencode_err = info.get("error")
                        if opencode_err:
                            err_data = opencode_err.get("data", {}) if isinstance(opencode_err, dict) else {}
                            err_msg = err_data.get("message") or str(opencode_err)
                            status = BenchmarkStatus.ERROR
                            status_code = err_data.get("statusCode", 400)
                            if status_code == 429:
                                status = BenchmarkStatus.RATE_LIMITED
                                timeout_reason = TimeoutType.RATE_LIMIT_429
                            else:
                                timeout_reason = TimeoutType.CLIENT_ERROR_4XX
                            error_message = f"Provider Error: {err_msg}"

                        toks = info.get("tokens", {})
                        if toks:
                            tokens_captured.input_tokens = toks.get("input", 0)
                            tokens_captured.output_tokens = toks.get("output", 0)
                            tokens_captured.reasoning_tokens = toks.get("reasoning", 0)
                            cache_t = toks.get("cache", {})
                            tokens_captured.cached_read_tokens = cache_t.get("read", 0)
                            tokens_captured.cached_write_tokens = cache_t.get("write", 0)

                        parts = resp_data.get("parts", [])
                        for part in parts:
                            p_text = part.get("text", "")
                            if on_chunk and p_text:
                                on_chunk(p_text)

                            p_time = part.get("time", {})
                            p_start = p_time.get("start")
                            p_end = p_time.get("end")
                            if p_start and t_first_token is None:
                                t_first_token = float(p_start)
                            if p_start and p_end:
                                dur = float(p_end) - float(p_start)
                                if dur > 0:
                                    inter_token_latencies.append(dur)

                        if t_first_token is None:
                            t_first_token = request_start_ms + (
                                (t_connected - t0) * 1000.0 if t_connected else 50.0
                            )

        except httpx.ConnectTimeout:
            timed_out = True
            status = BenchmarkStatus.TIMEOUT
            timeout_reason = TimeoutType.CONNECT_TIMEOUT
            error_message = "Connection timed out connecting to OpenCode server"
        except httpx.ReadTimeout:
            timed_out = True
            status = BenchmarkStatus.TIMEOUT
            timeout_reason = (
                TimeoutType.TTFT_TIMEOUT if t_first_token is None else TimeoutType.DEADLINE_TIMEOUT
            )
            error_message = "Read timed out waiting for OpenCode response"
        except Exception as exc:
            status = BenchmarkStatus.ERROR
            timeout_reason = TimeoutType.UNKNOWN_ERROR
            error_message = str(exc)

        completed_ms = time.time() * 1000.0
        timings = compute_timing_metrics(
            request_start_ms=request_start_ms,
            connection_ms=(t_connected - t0) * 1000.0 if t_connected else None,
            first_token_ms=t_first_token,
            completed_ms=completed_ms,
            inter_token_latencies_ms=inter_token_latencies,
        )

        tps = compute_tps_metrics(tokens_captured, timings)

        # Privacy cleanup: Delete session
        if self.auto_cleanup and session_id:
            await self.cleanup_session(session_id)

        return BenchmarkResult(
            id=req_id,
            provider="opencode-server",
            model=model or "default",
            status=status,
            timeout_type=timeout_reason,
            tokens=tokens_captured,
            timings=timings,
            tps=tps,
            error_message=error_message,
        )
