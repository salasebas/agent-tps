from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class TimeoutType(str, Enum):
    NONE = "none"
    CONNECT_TIMEOUT = "connect_timeout"
    TTFT_TIMEOUT = "ttft_timeout"
    STALL_TIMEOUT = "stall_timeout"
    DEADLINE_TIMEOUT = "deadline_timeout"
    RATE_LIMIT_429 = "rate_limit_429"
    SERVER_ERROR_5XX = "server_error_5xx"
    CLIENT_ERROR_4XX = "client_error_4xx"
    PROCESS_CRASH = "process_crash"
    UNKNOWN_ERROR = "unknown_error"


class BenchmarkStatus(str, Enum):
    SUCCESS = "success"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"


class TokenMetrics(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_read_tokens: int = 0
    cached_write_tokens: int = 0

    @property
    def generated_tokens(self) -> int:
        """Total newly generated tokens (output + reasoning)."""
        return self.output_tokens + self.reasoning_tokens

    @property
    def total_tokens(self) -> int:
        """Total tokens processed in the turn/session."""
        return self.input_tokens + self.output_tokens + self.reasoning_tokens + self.cached_write_tokens

    @property
    def cache_hit_rate(self) -> float:
        total_input = self.input_tokens + self.cached_read_tokens
        if total_input <= 0:
            return 0.0
        return self.cached_read_tokens / total_input


class TimingMetrics(BaseModel):
    request_start_ms: float
    connection_ms: float | None = None
    first_token_ms: float | None = None
    completed_ms: float | None = None

    ttft_ms: float | None = None
    generation_duration_ms: float | None = None
    total_duration_ms: float | None = None

    inter_token_latencies_ms: list[float] = Field(default_factory=list)
    itl_p50_ms: float = 0.0
    itl_p90_ms: float = 0.0
    itl_p95_ms: float = 0.0
    itl_p99_ms: float = 0.0
    itl_max_ms: float = 0.0
    jitter_ms: float = 0.0


class TPSMetrics(BaseModel):
    decode_tps: float = 0.0
    e2e_tps: float = 0.0
    total_throughput_tps: float = 0.0


class BenchmarkResult(BaseModel):
    id: str
    provider: str
    model: str
    status: BenchmarkStatus
    timeout_type: TimeoutType = TimeoutType.NONE
    tokens: TokenMetrics = Field(default_factory=TokenMetrics)
    timings: TimingMetrics
    tps: TPSMetrics = Field(default_factory=TPSMetrics)
    error_message: str | None = None
    retry_after_s: float | None = None
    raw_response_preview: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConcurrencyReport(BaseModel):
    concurrency_level: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    timed_out_requests: int
    rate_limited_requests: int
    wall_clock_duration_s: float
    aggregate_decode_tps: float
    aggregate_e2e_tps: float
    mean_ttft_ms: float
    p50_ttft_ms: float
    p95_ttft_ms: float
    p99_ttft_ms: float
    mean_worker_decode_tps: float
    degradation_percent: float = 0.0
    results: list[BenchmarkResult] = Field(default_factory=list)
