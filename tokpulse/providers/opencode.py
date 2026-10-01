from __future__ import annotations

import asyncio
from collections.abc import Callable
import contextlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time
from typing import Any

from tokpulse.core.calculator import compute_timing_metrics, compute_tps_metrics
from tokpulse.core.models import (
    BenchmarkResult,
    BenchmarkStatus,
    TimeoutType,
    TimingMetrics,
    TokenMetrics,
    TPSMetrics,
)
from tokpulse.providers.base import BaseAgentRunner


class OpenCodeSessionDetail:
    def __init__(
        self,
        session_id: str,
        title: str,
        provider: str,
        model: str,
        agent: str,
        time_created_ms: int,
        time_updated_ms: int,
        duration_ms: int,
        tokens: TokenMetrics,
        timings: TimingMetrics,
        tps: TPSMetrics,
        cost: float = 0.0,
        messages: list[dict[str, Any]] | None = None,
    ):
        self.session_id = session_id
        self.title = title
        self.provider = provider
        self.model = model
        self.agent = agent
        self.time_created_ms = time_created_ms
        self.time_updated_ms = time_updated_ms
        self.duration_ms = duration_ms
        self.tokens = tokens
        self.timings = timings
        self.tps = tps
        self.cost = cost
        self.messages = messages or []


class OpenCodeDBReader:
    """Reads session and execution telemetry directly from the OpenCode SQLite database."""

    DEFAULT_DB_PATH = Path.home() / ".local/share" / "opencode" / "opencode.db"

    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path) if db_path else self.DEFAULT_DB_PATH

    def is_available(self) -> bool:
        return self.db_path.exists() and self.db_path.is_file()

    def _get_connection(self, read_only: bool = True) -> sqlite3.Connection:
        if not self.is_available():
            raise FileNotFoundError(f"OpenCode database not found at {self.db_path}")
        if read_only:
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        else:
            conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 3000")
        return conn

    def get_recent_sessions(self, limit: int = 15) -> list[OpenCodeSessionDetail]:
        """Fetch the most recent sessions and calculate their performance metrics."""
        with self._get_connection(read_only=True) as conn:
            query = """
                SELECT
                    id, title, model, agent, cost,
                    tokens_input, tokens_output, tokens_reasoning,
                    tokens_cache_read, tokens_cache_write,
                    time_created, time_updated
                FROM session
                ORDER BY time_created DESC
                LIMIT ?
            """
            rows = conn.execute(query, (limit,)).fetchall()
            return [self._build_session_detail(conn, row) for row in rows]

    def _build_session_detail(
        self, conn: sqlite3.Connection, session_row: sqlite3.Row
    ) -> OpenCodeSessionDetail:
        session_id = session_row["id"]
        title = session_row["title"] or "Untitled"
        agent = session_row["agent"] or "build"
        cost = float(session_row["cost"] or 0.0)

        # Parse model column (stored as JSON string or raw slug)
        model_raw = session_row["model"] or ""
        provider_name = "opencode"
        model_name = "unknown"
        if model_raw:
            try:
                parsed_model = json.loads(model_raw)
                if isinstance(parsed_model, dict):
                    provider_name = parsed_model.get("providerID", "opencode")
                    model_name = parsed_model.get("id", parsed_model.get("modelID", "unknown"))
                else:
                    model_name = str(model_raw)
            except Exception:
                model_name = str(model_raw)
                if "/" in model_name:
                    provider_name, model_name = model_name.split("/", 1)

        tokens = TokenMetrics(
            input_tokens=int(session_row["tokens_input"] or 0),
            output_tokens=int(session_row["tokens_output"] or 0),
            reasoning_tokens=int(session_row["tokens_reasoning"] or 0),
            cached_read_tokens=int(session_row["tokens_cache_read"] or 0),
            cached_write_tokens=int(session_row["tokens_cache_write"] or 0),
        )

        time_created = int(session_row["time_created"])
        time_updated = int(session_row["time_updated"])
        total_duration_ms = max(0, time_updated - time_created)

        msg_rows = conn.execute(
            "SELECT id, time_created, time_updated, data FROM message WHERE session_id = ? ORDER BY time_created ASC",
            (session_id,),
        ).fetchall()

        messages_data: list[dict[str, Any]] = []
        first_token_ms: float | None = None
        last_token_ms: float | None = None
        gen_duration_accum_ms = 0.0
        step_durations: list[float] = []

        for m_row in msg_rows:
            try:
                m_data = json.loads(m_row["data"])
            except Exception:
                m_data = {}

            m_created = m_data.get("time", {}).get("created", m_row["time_created"])
            m_completed = m_data.get("time", {}).get("completed", m_row["time_updated"])
            m_role = m_data.get("role", "user")

            messages_data.append(
                {
                    "id": m_row["id"],
                    "role": m_role,
                    "time_created": m_created,
                    "time_completed": m_completed,
                }
            )

            part_rows = conn.execute(
                "SELECT id, time_created, data FROM part WHERE message_id = ? ORDER BY time_created ASC",
                (m_row["id"],),
            ).fetchall()

            for p_row in part_rows:
                try:
                    p_data = json.loads(p_row["data"])
                    p_time = p_data.get("time", {})
                    p_start = p_time.get("start")
                    p_end = p_time.get("end")
                    if p_start and p_end:
                        dur = float(p_end) - float(p_start)
                        if dur > 0:
                            step_durations.append(dur)
                            if first_token_ms is None or p_start < first_token_ms:
                                first_token_ms = float(p_start)
                            if last_token_ms is None or p_end > last_token_ms:
                                last_token_ms = float(p_end)
                            gen_duration_accum_ms += dur
                except Exception:
                    continue

        timings = compute_timing_metrics(
            request_start_ms=float(time_created),
            first_token_ms=first_token_ms,
            completed_ms=last_token_ms if last_token_ms is not None else float(time_updated),
            inter_token_latencies_ms=step_durations,
        )
        tps = compute_tps_metrics(tokens, timings)

        return OpenCodeSessionDetail(
            session_id=session_id,
            title=title,
            provider=provider_name,
            model=model_name,
            agent=agent,
            time_created_ms=time_created,
            time_updated_ms=time_updated,
            duration_ms=total_duration_ms,
            tokens=tokens,
            timings=timings,
            tps=tps,
            cost=cost,
            messages=messages_data,
        )

    def delete_session(self, session_id: str) -> bool:
        """Deletes a session and associated messages/parts from SQLite to maintain zero residual chats."""
        if not self.is_available() or not session_id:
            return False
        try:
            with self._get_connection(read_only=False) as conn:
                conn.execute("DELETE FROM part WHERE session_id = ?", (session_id,))
                conn.execute("DELETE FROM message WHERE session_id = ?", (session_id,))
                conn.execute("DELETE FROM session WHERE id = ?", (session_id,))
                conn.commit()
            return True
        except Exception:
            return False


class OpenCodeRunner(BaseAgentRunner):
    """Executes active benchmark prompts through OpenCode CLI with high-resolution streaming metrics."""

    def __init__(
        self,
        binary_path: str = "opencode",
        db_reader: OpenCodeDBReader | None = None,
        auto_cleanup: bool = True,
    ):
        self.binary_path = binary_path
        self.db_reader = db_reader or OpenCodeDBReader()
        self.auto_cleanup = auto_cleanup

    async def cleanup_session(self, session_id: str | None = None) -> None:
        """Wipes the created session from opencode.db to protect user privacy."""
        if session_id:
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self.db_reader.delete_session, session_id)
            except Exception:
                pass

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
        request_start_ms = time.time() * 1000.0

        if not shutil.which(self.binary_path):
            return BenchmarkResult(
                id=req_id,
                provider="opencode",
                model=model or "default",
                status=BenchmarkStatus.ERROR,
                timeout_type=TimeoutType.PROCESS_CRASH,
                timings=compute_timing_metrics(request_start_ms=request_start_ms),
                error_message=f"Executable '{self.binary_path}' not found in PATH.",
            )

        max_retries = 3
        last_attempt_data: dict[str, Any] = {}

        for attempt in range(max_retries):
            attempt_res = await self._run_single_attempt(
                cmd=cmd,
                ttft_timeout_s=ttft_timeout_s,
                stall_timeout_s=stall_timeout_s,
                deadline_timeout_s=deadline_timeout_s,
                on_chunk=on_chunk,
            )
            last_attempt_data = attempt_res
            stderr_str = attempt_res.get("stderr_str", "")

            # Check for sqlite lock contention
            if (
                "database is locked" in stderr_str.lower() or "sqlite_busy" in stderr_str.lower()
            ) and attempt < max_retries - 1:
                await asyncio.sleep(0.2 * (2**attempt))
                continue
            break

        completed_ms = time.time() * 1000.0
        tokens_captured = last_attempt_data.get("tokens", TokenMetrics())
        inter_token_latencies = last_attempt_data.get("latencies", [])
        t_first_token = last_attempt_data.get("t_first_token")
        captured_session_id = last_attempt_data.get("session_id")
        timed_out = last_attempt_data.get("timed_out", False)
        timeout_reason = last_attempt_data.get("timeout_reason", TimeoutType.NONE)
        error_message = last_attempt_data.get("error_message")

        timings = compute_timing_metrics(
            request_start_ms=request_start_ms,
            first_token_ms=t_first_token,
            completed_ms=completed_ms,
            inter_token_latencies_ms=inter_token_latencies,
        )

        if tokens_captured.output_tokens == 0 and len(inter_token_latencies) > 0:
            tokens_captured.output_tokens = len(inter_token_latencies) + 1

        tps = compute_tps_metrics(tokens_captured, timings)

        status = BenchmarkStatus.SUCCESS
        if timed_out:
            status = BenchmarkStatus.TIMEOUT
        elif timeout_reason == TimeoutType.RATE_LIMIT_429:
            status = BenchmarkStatus.RATE_LIMITED
        elif error_message:
            status = BenchmarkStatus.ERROR

        # Privacy cleanup: delete session from database
        if self.auto_cleanup and captured_session_id:
            await self.cleanup_session(captured_session_id)

        return BenchmarkResult(
            id=captured_session_id or req_id,
            provider="opencode",
            model=model or "default",
            status=status,
            timeout_type=timeout_reason,
            tokens=tokens_captured,
            timings=timings,
            tps=tps,
            error_message=error_message,
        )

    async def _run_single_attempt(
        self,
        cmd: list[str],
        ttft_timeout_s: float,
        stall_timeout_s: float,
        deadline_timeout_s: float,
        on_chunk: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Executes a single process invocation and parses JSON line events."""
        t0 = time.perf_counter()
        t_first_token: float | None = None
        t_last_token: float | None = None
        inter_token_latencies: list[float] = []
        tokens_captured = TokenMetrics()
        captured_session_id: str | None = None
        timed_out = False
        timeout_reason = TimeoutType.NONE
        error_message: str | None = None

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ.copy(),
        )

        async def read_stream():
            nonlocal t_first_token, t_last_token, captured_session_id
            if proc.stdout is None:
                return

            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                line_str = line.decode("utf-8", errors="replace").strip()
                if not line_str:
                    continue

                try:
                    event = json.loads(line_str)
                except json.JSONDecodeError:
                    continue

                now_perf = time.perf_counter()
                now_ms = time.time() * 1000.0

                if "sessionID" in event and not captured_session_id:
                    captured_session_id = event["sessionID"]

                evt_type = event.get("type")
                if evt_type == "step_start":
                    part = event.get("part", {})
                    if "sessionID" in part and not captured_session_id:
                        captured_session_id = part["sessionID"]

                elif evt_type in ("text", "reasoning"):
                    part = event.get("part", {})
                    text_chunk = part.get("text", "")
                    if "sessionID" in part and not captured_session_id:
                        captured_session_id = part["sessionID"]

                    if t_first_token is None:
                        t_first_token = now_ms
                    else:
                        assert t_last_token is not None
                        itl_ms = (now_perf - t_last_token) * 1000.0
                        inter_token_latencies.append(itl_ms)

                    t_last_token = now_perf
                    if on_chunk and text_chunk:
                        on_chunk(text_chunk)

                elif evt_type == "step_finish":
                    part = event.get("part", {})
                    tokens_info = part.get("tokens", {})
                    if tokens_info:
                        tokens_captured.input_tokens = tokens_info.get("input", tokens_captured.input_tokens)
                        tokens_captured.output_tokens = tokens_info.get(
                            "output", tokens_captured.output_tokens
                        )
                        tokens_captured.reasoning_tokens = tokens_info.get(
                            "reasoning", tokens_captured.reasoning_tokens
                        )
                        cache_info = tokens_info.get("cache", {})
                        tokens_captured.cached_read_tokens = cache_info.get(
                            "read", tokens_captured.cached_read_tokens
                        )
                        tokens_captured.cached_write_tokens = cache_info.get(
                            "write", tokens_captured.cached_write_tokens
                        )

        async def watchdog():
            nonlocal timed_out, timeout_reason
            while True:
                await asyncio.sleep(0.05)
                now_perf = time.perf_counter()
                elapsed = now_perf - t0

                if elapsed > deadline_timeout_s:
                    timed_out = True
                    timeout_reason = TimeoutType.DEADLINE_TIMEOUT
                    with contextlib.suppress(ProcessLookupError):
                        proc.kill()
                    break

                if t_first_token is None:
                    if elapsed > ttft_timeout_s:
                        timed_out = True
                        timeout_reason = TimeoutType.TTFT_TIMEOUT
                        with contextlib.suppress(ProcessLookupError):
                            proc.kill()
                        break
                else:
                    if t_last_token is not None:
                        time_since_last = now_perf - t_last_token
                        if time_since_last > stall_timeout_s:
                            timed_out = True
                            timeout_reason = TimeoutType.STALL_TIMEOUT
                            with contextlib.suppress(ProcessLookupError):
                                proc.kill()
                            break

        stream_task = asyncio.create_task(read_stream())
        watchdog_task = asyncio.create_task(watchdog())

        await stream_task
        watchdog_task.cancel()

        stderr_bytes = await proc.stderr.read() if proc.stderr else b""
        await proc.wait()

        stderr_str = stderr_bytes.decode("utf-8", errors="replace")
        if "429" in stderr_str or "rate limit" in stderr_str.lower():
            timeout_reason = TimeoutType.RATE_LIMIT_429
            error_message = stderr_str.strip()

        if proc.returncode != 0 and not timed_out:
            error_message = stderr_str.strip() or f"Process exited with code {proc.returncode}"

        return {
            "tokens": tokens_captured,
            "latencies": inter_token_latencies,
            "t_first_token": t_first_token,
            "session_id": captured_session_id,
            "timed_out": timed_out,
            "timeout_reason": timeout_reason,
            "error_message": error_message,
            "stderr_str": stderr_str,
        }
