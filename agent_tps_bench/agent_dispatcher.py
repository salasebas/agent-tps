from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from typing import Any, Callable

from agent_tps_bench.calculator import compute_timing_metrics, compute_tps_metrics
from agent_tps_bench.llm_stream_runner import LLMStreamRunner
from agent_tps_bench.models import (
    BenchmarkResult,
    BenchmarkStatus,
    TimeoutType,
    TimingMetrics,
    TokenMetrics,
)
from agent_tps_bench.opencode_runner import OpenCodeRunner
from agent_tps_bench.opencode_server_runner import OpenCodeServerRunner


class GenericCliAgentRunner:
    """Runs a standard CLI agent (Claude Code, Codex, Grok, Cursor) and extracts timing and tokens."""

    def __init__(self, provider: str, binary_path: str):
        self.provider = provider
        self.binary_path = binary_path

    async def run_prompt(
        self,
        prompt: str,
        model: str | None = None,
        deadline_timeout_s: float = 60.0,
        ttft_timeout_s: float = 15.0,
        stall_timeout_s: float = 10.0,
        on_chunk: Callable[[str], None] | None = None,
        **kwargs: Any,
    ) -> BenchmarkResult:
        req_id = f"{self.provider}_{int(time.time() * 1000)}"
        t0 = time.perf_counter()
        request_start_ms = time.time() * 1000.0

        if not shutil.which(self.binary_path):
            return BenchmarkResult(
                id=req_id,
                provider=self.provider,
                model=model or "default",
                status=BenchmarkStatus.ERROR,
                timeout_type=TimeoutType.PROCESS_CRASH,
                error_message=f"CLI '{self.binary_path}' is not installed or not in PATH.",
                timings=TimingMetrics(request_start_ms=request_start_ms),
            )

        cmd = [self.binary_path]
        if self.provider == "claude":
            cmd.extend(["-p", "--output-format", "json"])
            if model:
                cmd.extend(["--fallback-model", model])
            cmd.append(prompt)
        elif self.provider == "codex":
            cmd.extend(["exec"])
            if model:
                cmd.extend(["-c", f"model='{model}'"])
            cmd.append(prompt)
        elif self.provider == "grok":
            cmd.append(prompt)
        elif self.provider == "cursor":
            cmd.append(prompt)
        else:
            cmd.append(prompt)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=os.environ.copy(),
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=deadline_timeout_s,
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except Exception:
                pass
            return BenchmarkResult(
                id=req_id,
                provider=self.provider,
                model=model or "default",
                status=BenchmarkStatus.TIMEOUT,
                timeout_type=TimeoutType.DEADLINE_TIMEOUT,
                error_message=f"Agent exceeded deadline timeout of {deadline_timeout_s}s",
                timings=TimingMetrics(
                    request_start_ms=request_start_ms,
                    completed_ms=request_start_ms + (deadline_timeout_s * 1000.0),
                ),
            )
        except Exception as e:
            return BenchmarkResult(
                id=req_id,
                provider=self.provider,
                model=model or "default",
                status=BenchmarkStatus.ERROR,
                timeout_type=TimeoutType.UNKNOWN_ERROR,
                error_message=str(e),
                timings=TimingMetrics(request_start_ms=request_start_ms),
            )

        t_end = time.perf_counter()
        total_duration_ms = (t_end - t0) * 1000.0
        stdout_str = stdout_bytes.decode("utf-8", errors="replace").strip()
        stderr_str = stderr_bytes.decode("utf-8", errors="replace").strip()

        tokens = TokenMetrics()
        response_preview = ""
        ttft_ms = total_duration_ms * 0.6  # Default prefill estimate for monolithic output

        # Parse Claude JSON format if provider is claude
        if self.provider == "claude" and stdout_str.startswith("{"):
            try:
                parsed = json.loads(stdout_str)
                if parsed.get("is_error"):
                    return BenchmarkResult(
                        id=req_id,
                        provider=self.provider,
                        model=model or "default",
                        status=BenchmarkStatus.ERROR,
                        timeout_type=TimeoutType.PROCESS_CRASH,
                        error_message=parsed.get("result", "Unknown Claude CLI error"),
                        timings=TimingMetrics(request_start_ms=request_start_ms, completed_ms=request_start_ms + total_duration_ms),
                    )
                usage = parsed.get("usage", {})
                tokens.input_tokens = usage.get("input_tokens", 0)
                tokens.output_tokens = usage.get("output_tokens", 0)
                tokens.cached_read_tokens = usage.get("cache_read_input_tokens", 0)
                tokens.cached_write_tokens = usage.get("cache_creation_input_tokens", 0)
                response_preview = str(parsed.get("result", ""))
                duration_api_ms = parsed.get("duration_api_ms")
                if duration_api_ms and duration_api_ms > 0:
                    total_duration_ms = float(duration_api_ms)
            except Exception:
                pass

        if tokens.output_tokens == 0:
            # Fallback estimation based on text length
            text_to_eval = response_preview or stdout_str
            tokens.output_tokens = max(1, len(text_to_eval) // 4)

        if proc.returncode != 0 and not response_preview:
            status = BenchmarkStatus.ERROR
            error_msg = stderr_str or stdout_str or f"Process exited with code {proc.returncode}"
            return BenchmarkResult(
                id=req_id,
                provider=self.provider,
                model=model or "default",
                status=status,
                timeout_type=TimeoutType.PROCESS_CRASH,
                error_message=error_msg,
                timings=TimingMetrics(request_start_ms=request_start_ms, completed_ms=request_start_ms + total_duration_ms),
            )

        timings = compute_timing_metrics(
            request_start_ms=request_start_ms,
            first_token_ms=request_start_ms + ttft_ms,
            completed_ms=request_start_ms + total_duration_ms,
        )
        tps = compute_tps_metrics(tokens, timings)

        return BenchmarkResult(
            id=req_id,
            provider=self.provider,
            model=model or "default",
            status=BenchmarkStatus.SUCCESS,
            tokens=tokens,
            timings=timings,
            tps=tps,
            raw_response_preview=(response_preview or stdout_str)[:200],
        )


def get_runner_for_provider(
    provider: str,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Any:
    """Factory creating the appropriate runner based on the chosen provider."""
    p = provider.lower()
    if p == "opencode":
        return OpenCodeRunner()
    elif p == "opencode-server":
        return OpenCodeServerRunner()
    elif p in ("claude", "claude-code"):
        return GenericCliAgentRunner("claude", "claude")
    elif p == "codex":
        return GenericCliAgentRunner("codex", "codex")
    elif p == "grok":
        return GenericCliAgentRunner("grok", "grok")
    elif p == "cursor":
        return GenericCliAgentRunner("cursor", "cursor")
    elif p == "antigravity":
        # Google Antigravity / Gemini via LLMStreamRunner or agy CLI
        gemini_key = api_key or os.environ.get("GEMINI_API_KEY")
        return LLMStreamRunner(
            base_url=base_url or "https://generativelanguage.googleapis.com/v1beta/openai",
            api_key=gemini_key,
            provider="antigravity",
        )
    else:
        # Direct API providers: groq, cerebras, openrouter, openai, ollama
        env_keys = {
            "groq": "GROQ_API_KEY",
            "cerebras": "CEREBRAS_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "openai": "OPENAI_API_KEY",
        }
        resolved_key = api_key or (os.environ.get(env_keys[p]) if p in env_keys else None)
        return LLMStreamRunner(base_url=base_url, api_key=resolved_key, provider=p)
