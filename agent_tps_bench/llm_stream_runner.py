from __future__ import annotations

import asyncio
import json
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


class LLMStreamRunner:
    """Benchmarks direct LLM provider endpoints with millisecond-precision streaming and watchdog timeouts."""

    PROVIDER_PRESETS: dict[str, str] = {
        "openai": "https://api.openai.com/v1",
        "openrouter": "https://openrouter.ai/api/v1",
        "groq": "https://api.groq.com/openai/v1",
        "cerebras": "https://api.cerebras.ai/v1",
        "deepseek": "https://api.deepseek.com",
        "ollama": "http://localhost:11434/v1",
    }

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        provider: str = "custom",
    ):
        self.provider = provider.lower()
        if base_url:
            self.base_url = base_url.rstrip("/")
        elif self.provider in self.PROVIDER_PRESETS:
            self.base_url = self.PROVIDER_PRESETS[self.provider]
        else:
            self.base_url = "https://api.openai.com/v1"

        self.api_key = api_key or ""

    async def run_stream(
        self,
        model: str,
        prompt: str,
        system_prompt: str | None = None,
        connect_timeout_s: float = 8.0,
        ttft_timeout_s: float = 12.0,
        stall_timeout_s: float = 5.0,
        deadline_timeout_s: float = 60.0,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        extra_headers: dict[str, str] | None = None,
        on_chunk: Callable[[str], None] | None = None,
    ) -> BenchmarkResult:
        """Executes a streaming request against the LLM provider and measures timing, TPS, and timeouts."""
        req_id = f"stream_{int(time.time() * 1000)}"
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "User-Agent": "agent-tps-bench/0.1.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if extra_headers:
            headers.update(extra_headers)

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
        retry_after_s: float | None = None

        timeout_config = httpx.Timeout(
            connect=connect_timeout_s,
            read=deadline_timeout_s,
            write=connect_timeout_s,
            pool=connect_timeout_s,
        )

        last_activity_time = t0
        received_any_content = False

        try:
            async with httpx.AsyncClient(timeout=timeout_config) as client:
                url = f"{self.base_url}/chat/completions"
                # Connect phase
                try:
                    async with client.stream("POST", url, json=payload, headers=headers) as response:
                        t_connected = time.perf_counter()
                        last_activity_time = t_connected

                        # Check HTTP status codes
                        if response.status_code == 429:
                            error_body = (await response.aread()).decode("utf-8", errors="replace")
                            status = BenchmarkStatus.RATE_LIMITED
                            timeout_reason = TimeoutType.RATE_LIMIT_429
                            retry_hdr = response.headers.get("retry-after")
                            if retry_hdr:
                                try:
                                    retry_after_s = float(retry_hdr)
                                except ValueError:
                                    retry_after_s = None
                            error_message = f"Rate limited (429): {error_body[:200]}"
                            t_end = time.perf_counter()
                            return self._build_result(
                                req_id, model, status, timeout_reason, tokens_captured,
                                request_start_ms, t0, t_connected, None, t_end,
                                inter_token_latencies, error_message, retry_after_s, ""
                            )

                        if response.status_code >= 500:
                            error_body = (await response.aread()).decode("utf-8", errors="replace")
                            status = BenchmarkStatus.ERROR
                            timeout_reason = TimeoutType.SERVER_ERROR_5XX
                            error_message = f"Server Error ({response.status_code}): {error_body[:200]}"
                            t_end = time.perf_counter()
                            return self._build_result(
                                req_id, model, status, timeout_reason, tokens_captured,
                                request_start_ms, t0, t_connected, None, t_end,
                                inter_token_latencies, error_message, retry_after_s, ""
                            )

                        if response.status_code != 200:
                            error_body = (await response.aread()).decode("utf-8", errors="replace")
                            status = BenchmarkStatus.ERROR
                            timeout_reason = TimeoutType.CLIENT_ERROR_4XX
                            error_message = f"HTTP Error ({response.status_code}): {error_body[:200]}"
                            t_end = time.perf_counter()
                            return self._build_result(
                                req_id, model, status, timeout_reason, tokens_captured,
                                request_start_ms, t0, t_connected, None, t_end,
                                inter_token_latencies, error_message, retry_after_s, ""
                            )

                        # Stream reading loop with watchdog
                        line_iterator = response.aiter_lines()
                        while True:
                            now = time.perf_counter()
                            deadline_remaining = deadline_timeout_s - (now - t0)
                            if deadline_remaining <= 0:
                                timed_out = True
                                timeout_reason = TimeoutType.DEADLINE_TIMEOUT
                                error_message = f"Deadline exceeded: {deadline_timeout_s}s"
                                break

                            if not received_any_content:
                                wait_timeout = ttft_timeout_s - (now - t0)
                                if wait_timeout <= 0:
                                    timed_out = True
                                    timeout_reason = TimeoutType.TTFT_TIMEOUT
                                    error_message = f"TTFT timeout exceeded: {ttft_timeout_s}s"
                                    break
                            else:
                                wait_timeout = min(stall_timeout_s, deadline_remaining)

                            try:
                                line = await asyncio.wait_for(
                                    line_iterator.__anext__(),
                                    timeout=max(0.001, wait_timeout),
                                )
                            except StopAsyncIteration:
                                break
                            except asyncio.TimeoutError:
                                timed_out = True
                                if not received_any_content:
                                    timeout_reason = TimeoutType.TTFT_TIMEOUT
                                    error_message = f"TTFT timeout exceeded: {ttft_timeout_s}s"
                                else:
                                    timeout_reason = TimeoutType.STALL_TIMEOUT
                                    error_message = f"Stall timeout: no token for {stall_timeout_s}s"
                                break

                            now = time.perf_counter()

                            if not line or not line.startswith("data: "):
                                continue

                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                break

                            try:
                                chunk = json.loads(data_str)
                            except Exception:
                                continue

                            # Inspect usage object
                            usage = chunk.get("usage")
                            if usage:
                                prompt_tokens = usage.get("prompt_tokens", 0)
                                completion_tokens = usage.get("completion_tokens", 0)
                                reasoning_tokens = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
                                cached_read = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
                                tokens_captured = TokenMetrics(
                                    input_tokens=prompt_tokens,
                                    output_tokens=max(0, completion_tokens - reasoning_tokens),
                                    reasoning_tokens=reasoning_tokens,
                                    cached_read_tokens=cached_read,
                                )

                            # Inspect content delta
                            choices = chunk.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                text_chunk = delta.get("content") or delta.get("reasoning_content") or ""
                                if text_chunk:
                                    text_preview_parts.append(text_chunk)
                                    if on_chunk:
                                        on_chunk(text_chunk)

                                    if not received_any_content:
                                        received_any_content = True
                                        t_first_token = now
                                        last_activity_time = now
                                    else:
                                        delta_ms = (now - last_activity_time) * 1000.0
                                        inter_token_latencies.append(delta_ms)
                                        last_activity_time = now

                                    t_last_token = now

                except httpx.ConnectTimeout:
                    timed_out = True
                    timeout_reason = TimeoutType.CONNECT_TIMEOUT
                    error_message = f"Connect timeout exceeded ({connect_timeout_s}s)"
                except httpx.ReadTimeout:
                    timed_out = True
                    if not received_any_content:
                        timeout_reason = TimeoutType.TTFT_TIMEOUT
                        error_message = "TTFT read timeout: no first token within threshold"
                    else:
                        timeout_reason = TimeoutType.STALL_TIMEOUT
                        error_message = f"Stall timeout: no token received for {stall_timeout_s}s"

        except Exception as e:
            if not timed_out:
                status = BenchmarkStatus.ERROR
                timeout_reason = TimeoutType.UNKNOWN_ERROR
                error_message = str(e)

        t_end = time.perf_counter()

        if timed_out:
            status = BenchmarkStatus.TIMEOUT
        elif not received_any_content and status == BenchmarkStatus.SUCCESS:
            status = BenchmarkStatus.ERROR
            timeout_reason = TimeoutType.CLIENT_ERROR_4XX
            error_message = "Stream ended without any content"

        # Fallback token estimation if usage was not reported by provider
        if tokens_captured.output_tokens == 0 and text_preview_parts:
            full_text = "".join(text_preview_parts)
            tokens_captured.output_tokens = max(1, len(full_text) // 4)

        return self._build_result(
            req_id, model, status, timeout_reason, tokens_captured,
            request_start_ms, t0, t_connected, t_first_token, t_end,
            inter_token_latencies, error_message, retry_after_s, "".join(text_preview_parts)
        )

    def _build_result(
        self,
        req_id: str,
        model: str,
        status: BenchmarkStatus,
        timeout_reason: TimeoutType,
        tokens_captured: TokenMetrics,
        request_start_ms: float,
        t0: float,
        t_connected: float | None,
        t_first_token: float | None,
        t_end: float,
        inter_token_latencies: list[float],
        error_message: str | None,
        retry_after_s: float | None,
        preview_text: str,
    ) -> BenchmarkResult:
        connection_ms = (
            request_start_ms + ((t_connected - t0) * 1000.0)
            if t_connected is not None
            else None
        )
        first_token_ms = (
            request_start_ms + ((t_first_token - t0) * 1000.0)
            if t_first_token is not None
            else None
        )
        completed_ms = request_start_ms + ((t_end - t0) * 1000.0)

        timings = compute_timing_metrics(
            request_start_ms=request_start_ms,
            first_token_ms=first_token_ms,
            completed_ms=completed_ms,
            inter_token_latencies_ms=inter_token_latencies,
            connection_ms=connection_ms,
        )

        tps = compute_tps_metrics(tokens_captured, timings)

        return BenchmarkResult(
            id=req_id,
            provider=self.provider,
            model=model,
            status=status,
            timeout_type=timeout_reason,
            tokens=tokens_captured,
            timings=timings,
            tps=tps,
            error_message=error_message,
            retry_after_s=retry_after_s,
            raw_response_preview=preview_text[:200],
            metadata={"base_url": self.base_url},
        )
