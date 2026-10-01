from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any
from pydantic import BaseModel, Field

from agent_tps_bench.calculator import compute_timing_metrics, compute_tps_metrics
from agent_tps_bench.models import (
    BenchmarkResult,
    BenchmarkStatus,
    TimeoutType,
    TimingMetrics,
    TokenMetrics,
    TPSMetrics,
)


class OpenCodeSessionDetail(BaseModel):
    session_id: str
    title: str
    provider: str
    model: str
    agent: str
    time_created_ms: int
    time_updated_ms: int
    duration_ms: int
    tokens: TokenMetrics
    timings: TimingMetrics
    tps: TPSMetrics
    cost: float = 0.0
    messages: list[dict[str, Any]] = Field(default_factory=list)


class OpenCodeDBReader:
    """Reads session and execution telemetry directly from the OpenCode SQLite database."""

    DEFAULT_DB_PATH = Path.home() / ".local/share" / "opencode" / "opencode.db"

    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path) if db_path else self.DEFAULT_DB_PATH

    def is_available(self) -> bool:
        return self.db_path.exists() and self.db_path.is_file()

    def _get_connection(self) -> sqlite3.Connection:
        if not self.is_available():
            raise FileNotFoundError(f"OpenCode database not found at {self.db_path}")
        # Read-only URI connection to prevent locking the database from live OpenCode runs
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 2000")
        return conn

    def get_recent_sessions(self, limit: int = 15) -> list[OpenCodeSessionDetail]:
        """Fetch the most recent sessions and calculate their performance metrics."""
        with self._get_connection() as conn:
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
            results: list[OpenCodeSessionDetail] = []
            for row in rows:
                session_id = row["id"]
                detail = self._build_session_detail(conn, row)
                results.append(detail)
            return results

    def get_session_by_id(self, session_id: str) -> OpenCodeSessionDetail | None:
        """Fetch a specific session by its session ID."""
        with self._get_connection() as conn:
            query = """
                SELECT 
                    id, title, model, agent, cost,
                    tokens_input, tokens_output, tokens_reasoning,
                    tokens_cache_read, tokens_cache_write,
                    time_created, time_updated
                FROM session
                WHERE id = ?
            """
            row = conn.execute(query, (session_id,)).fetchone()
            if not row:
                return None
            return self._build_session_detail(conn, row)

    def _build_session_detail(self, conn: sqlite3.Connection, session_row: sqlite3.Row) -> OpenCodeSessionDetail:
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

        # Retrieve messages and parts to inspect finer-grained timings
        msg_rows = conn.execute(
            "SELECT id, time_created, time_updated, data FROM message WHERE session_id = ? ORDER BY time_created ASC",
            (session_id,),
        ).fetchall()

        messages_data: list[dict[str, Any]] = []
        first_token_ms: float | None = None
        last_token_ms: float | None = None
        gen_duration_accum_ms = 0.0

        for m_row in msg_rows:
            try:
                m_data = json.loads(m_row["data"])
            except Exception:
                m_data = {}

            m_created = m_data.get("time", {}).get("created", m_row["time_created"])
            m_completed = m_data.get("time", {}).get("completed", m_row["time_updated"])
            m_role = m_data.get("role", "user")

            messages_data.append({
                "id": m_row["id"],
                "role": m_role,
                "time_created": m_created,
                "time_completed": m_completed,
                "tokens": m_data.get("tokens", {}),
            })

            # Check parts for this message
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
                        if first_token_ms is None or p_start < first_token_ms:
                            first_token_ms = float(p_start)
                        if last_token_ms is None or p_end > last_token_ms:
                            last_token_ms = float(p_end)
                        gen_duration_accum_ms += max(0.0, float(p_end - p_start))
                except Exception:
                    continue

            # Fallback to message level completed - created if parts don't carry individual spans
            if gen_duration_accum_ms == 0.0 and m_role == "assistant" and m_completed and m_created:
                diff = float(m_completed - m_created)
                if diff > 0:
                    gen_duration_accum_ms += diff

        # If we have finer timing from parts/messages, use them
        if first_token_ms is None:
            # Estimate first token at session creation + estimated prefill
            first_token_ms = float(time_created)
        if last_token_ms is None:
            last_token_ms = float(time_updated)

        timings = compute_timing_metrics(
            request_start_ms=float(time_created),
            first_token_ms=first_token_ms,
            completed_ms=float(time_updated),
            inter_token_latencies_ms=[],
        )
        if gen_duration_accum_ms > 0:
            timings.generation_duration_ms = gen_duration_accum_ms

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

    def to_benchmark_result(self, detail: OpenCodeSessionDetail) -> BenchmarkResult:
        """Convert a session detail into a normalized BenchmarkResult."""
        return BenchmarkResult(
            id=detail.session_id,
            provider=detail.provider,
            model=detail.model,
            status=BenchmarkStatus.SUCCESS,
            timeout_type=TimeoutType.NONE,
            tokens=detail.tokens,
            timings=detail.timings,
            tps=detail.tps,
            metadata={
                "title": detail.title,
                "agent": detail.agent,
                "cost_usd": detail.cost,
            },
        )
