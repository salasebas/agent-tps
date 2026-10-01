# agent-tps-bench

> High-precision benchmark suite for Agent and LLM **Tokens Per Second (TPS)**, **Time to First Token (TTFT)**, **Timeout Classification**, **Concurrency Degradation**, and **Rate Limits**.

Inspired by agent harness architectures like [T3 Code](https://github.com/pingdotgg/t3code), this tool bridges the gap between **raw LLM streaming inference** and **multi-turn subagent execution**.

---

## Key Features

1. **Dual-Tier Benchmarking**:
   - **Agent Harness Tier (OpenCode)**: Executes real subagents via OpenCode CLI or connects to `opencode serve`. Automatically extracts SQLite session records from `~/.local/share/opencode/opencode.db` to cross-verify ground-truth token accounting (input, output, reasoning, cache read/write).
   - **Direct Provider Tier (LLM Streaming SSE)**: Probes OpenAI, OpenRouter, Groq, Cerebras, DeepSeek, and local Ollama endpoints with sub-millisecond precision.

2. **Mathematical Precision**:
   - **Decode TPS**: $\frac{\text{output\_tokens} + \text{reasoning\_tokens}}{\text{generation\_duration}}$ (true GPU inference speed).
   - **End-to-End TPS**: $\frac{\text{output\_tokens} + \text{reasoning\_tokens}}{\text{total\_wall\_clock\_duration}}$ (user-perceived speed).
   - **Total Throughput**: $\frac{\text{total\_tokens}}{\text{total\_duration}}$ (eval + generation throughput).
   - **Inter-Token Latency (ITL)**: Median (p50), tail (p90, p95, p99), Max stall, and RFC 3550 Jitter.
   - **Cache Efficiency**: Cache hit rate percentage and cached tokens read/written.

3. **Fine-Grained Timeout & Error Classification**:
   - `CONNECT_TIMEOUT`: Server connection could not be established.
   - `TTFT_TIMEOUT`: Model prefill/queue exceeded the initial token threshold.
   - `STALL_TIMEOUT`: Mid-stream heartbeat watchdog triggered when tokens paused.
   - `DEADLINE_TIMEOUT`: Total execution exceeded maximum deadline.
   - `RATE_LIMIT_429`: HTTP 429 Too Many Requests detected, parsing `Retry-After` backoff.
   - `SERVER_ERROR_5XX` & `CLIENT_ERROR_4XX`: Protocol-level failures.
   - `PROCESS_CRASH`: Subagent process terminated with non-zero exit code (e.g. SQLite locking).

4. **Subagent Concurrency & Stress Testing**:
   - Simulate $N$ concurrent workers.
   - Trace throughput saturation, concurrency degradation curves, and rate limit thresholds.
   - Export reports to JSON or Markdown.

---

## Architecture

```
                    ┌───────────────────────────────────┐
                    │            CLI & Runner           │
                    │  (tps-bench / Typer / Rich UI)    │
                    └─────────────────┬─────────────────┘
                                      │
              ┌───────────────────────┴───────────────────────┐
              ▼                                               ▼
┌───────────────────────────────┐               ┌───────────────────────────────┐
│       Agent Tier              │               │       Provider Tier           │
│   (OpenCode CLI / Server)     │               │   (OpenAI / Groq / Ollama)    │
└──────────────┬────────────────┘               └──────────────┬────────────────┘
               │                                               │
     ┌─────────┴─────────┐                                     │
     ▼                   ▼                                     ▼
┌──────────────┐  ┌─────────────┐                      ┌───────────────┐
│ opencode.db  │  │ opencode run│                      │  SSE Stream   │
│ SQLite Read  │  │ JSON Events │                      │  Watchdogs    │
└──────────────┘  └─────────────┘                      └───────────────┘
```

---

## Installation & Requirements

* macOS / Linux
* Python $\ge 3.13$
* [uv](https://github.com/astral-sh/uv) (recommended) or standard `pip`

```bash
cd ~/projects/agent-tps-bench
uv sync
```

---

## CLI Usage

### 1. Active OpenCode Benchmark
Launch a prompt through OpenCode and measure real-time streaming TPS, TTFT, and timeout classification:

```bash
uv run tps-bench opencode-bench --prompt "Explica que es un puntero en C en 2 lineas"
```

Configurable timeouts:
```bash
uv run tps-bench opencode-bench \
  -p "Calcula la serie de fibonacci" \
  --ttft-timeout 10.0 \
  --stall-timeout 4.0 \
  --deadline-timeout 30.0 \
  -o report.json
```

### 2. Historical OpenCode Session Telemetry
Inspect previously executed OpenCode sessions directly from the local database:

```bash
# View recent 10 sessions with calculated TPS and token breakdown
uv run tps-bench opencode-history --limit 10

# Inspect a specific session in detail
uv run tps-bench opencode-history --session ses_f0b6344d9ffeJ6cXqo46m5Goq0
```

### 3. Direct LLM Provider Streaming Benchmark
Test any OpenAI-compatible streaming API (Groq, OpenRouter, Cerebras, OpenAI, Ollama):

```bash
# Benchmark Groq
uv run tps-bench stream-bench \
  --provider groq \
  --model llama-3.3-70b-versatile \
  --api-key "$GROQ_API_KEY" \
  --prompt "Cuenta una historia corta de 3 parrafos"

# Benchmark local Ollama
uv run tps-bench stream-bench \
  --base-url "http://localhost:11434/v1" \
  --model "qwen2.5-coder:7b" \
  --prompt "Write a binary search function in Rust"
```

### 4. Subagent Concurrency & Stress Testing
Run parallel subagents and analyze concurrency degradation and rate limits:

```bash
# Run 4 concurrent subagents executing 8 total requests
uv run tps-bench stress --target opencode --concurrency 4 --total 8

# Concurrency sweep (1, 2, 4, 8 workers) to chart degradation curve
uv run tps-bench stress --target opencode --sweep --total 4 -o sweep.json
```

---

## Running the Test Suite

```bash
uv run pytest -v
```

All 16 unit and integration tests cover:
- Metric and percentile calculations (ITL, Jitter, Decode TPS, E2E TPS).
- OpenCode SQLite session reader and ground-truth validation.
- Mock OpenCode streaming event parser (`step_start`, `text`, `reasoning`, `step_finish`).
- Watchdog timeouts (`TTFT_TIMEOUT`, `STALL_TIMEOUT`, `DEADLINE_TIMEOUT`).
- HTTP 429 rate limit detection and backoff extraction.
- Concurrency orchestrator and degradation tracking.
