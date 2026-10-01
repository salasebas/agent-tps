<div align="center">

# ⚡ Agent-TPS

**Command-line TPS, latency, and concurrency profiler for AI coding agents**

[![CI](https://github.com/salasebas/agent-tps/actions/workflows/ci.yml/badge.svg)](https://github.com/salasebas/agent-tps/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-%3E%3D3.13-blue.svg)](https://www.python.org/)
[![Code Style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*Measure real-world Decode TPS, Time-to-First-Token (TTFT), Subagent Concurrency Scaling, and Failure Classifications across OpenCode, Cursor, Grok, Antigravity, Codex, and Claude Code.*

---

</div>

## 💡 Why Agent-TPS?

Standard LLM benchmark tools measure simple HTTP API completions. But modern developers work with **autonomous coding agents** that run in background terminal processes, execute multi-turn loops, inspect repository workspaces, and query local SQLite caches.

When evaluating coding agents, you need answers to critical questions:
- **What is the true generation velocity (Decode TPS)?** (Excluding initial prompt overhead & context serialization).
- **What is the real perceived latency (TTFT & E2E TPS)?**
- **How well does prompt caching perform?** (Prompt cache read vs write accounting).
- **Does throughput collapse under multi-agent concurrency?** (Subagents competing for local processes, locks, or hitting HTTP 429 rate limits).

**Agent-TPS** provides a unified speedometer, interactive terminal explorer, and multi-agent load tester designed specifically for developer coding agents.

---

## 🚀 Quickstart

Run Agent-TPS directly using [`uv`](https://github.com/astral-sh/uv):

```bash
# Launch the interactive terminal UI
uvx agent-tps

# Or run with the short alias
uvx atps
```

Alternatively, clone the repository for local development:

```bash
git clone https://github.com/salasebas/agent-tps.git
cd agent-tps
uv sync --extra dev
uv run agent-tps
```

---

## 🕹️ Interactive Terminal Interface (TUI)

Simply type `agent-tps` (or `atps`) with no arguments to open the clean interactive menu:

```
┌────────────────────────────────────────────────────────────────┐
│  ⚡ Agent-TPS v0.2.0                                            │
│  Coding Agent Velocity & Concurrency Profiler                  │
│  Engines: OpenCode · Cursor · Grok · Antigravity · Codex · Claude │
└────────────────────────────────────────────────────────────────┘

? What would you like to do?
 ❯ ⚡ Run Benchmark (Prompt, Model Fuzzy Finder, Live Speedometer)
   🔥 Subagent Stress Test (Concurrency & Rate Limits)
   📜 Saved Reports (Dynamic Standalone Inspector)
   🗑️ Clear / Delete Reports
   📂 OpenCode Local History (SQLite Telemetry)
   🚪 Exit
```

### Key Interactive Features:
1. **Global Fuzzy Model Finder**: Search across all agent models instantly (e.g. type `fable`, `gpt-6`, `gemini-2.5`, `longcat`, `composer`) without having to pick a provider first.
2. **Live Token Speedometer**: Live streaming token preview with real-time token counter.
3. **Dynamic Standalone Report Inspector**: Browse your saved benchmark runs and view full standalone performance telemetry cards on screen.
4. **Subagents Concurrency Tester**: Spin up parallel worker pools to test degradation under load and detect rate limits (HTTP 429).

---

## 🛡️ Supported Agent Engines

Agent-TPS natively integrates with the core agent CLI drivers:

| Agent Engine | Driver Binary | Default Model | Typical Execution Profile |
| :--- | :--- | :--- | :--- |
| **OpenCode** | `opencode` | `opencode/longcat-2.5-preview-free` | Local CLI JSON streaming, server mode (`opencode serve`), SQLite session tracking |
| **Cursor Agent** | `cursor-agent` / `cursor` | `auto` / `composer-2` | Specialized agentic composer with multi-file diff awareness |
| **Grok Agent** | `grok-build` / `grok` | `grok-build` | Terminal coding agent powered by xAI Grok 3 reasoning |
| **Antigravity** | `agy` / `antigravity` | `antigravity-default` | Deep multi-step reasoning agent with Gemini 2.5 Pro / Flash |
| **Codex** | `codex` | `gpt-6-astra` | OpenAI Codex CLI execution engine (`codex exec`) |
| **Claude Code** | `claude` | `claude-fable-5-1` | Anthropic's developer-focused CLI agent with structured JSON outputs |

---

## 🔒 Privacy Guarantee: Zero Chat Logging

Agent-TPS is built strictly for **performance benchmarking**, not surveillance:
- **No Chat History Stored**: Prompts and generated text are never written to disk in persistent benchmark reports.
- **Pure Numerical Telemetry**: Only performance metrics (tokens, TTFT, TPS, latencies, error types, timestamps) are saved.
- **Automatic Session Purge**: After every benchmark execution, temporary sessions and state files are immediately wiped from the agent host (including OpenCode SQLite and temporary scratch workspaces).

Metrics are stored locally on your machine in standard OS data paths:
- **macOS**: `~/Library/Application Support/agent-tps/runs/`
- **Linux**: `~/.local/share/agent-tps/runs/` (or `$XDG_DATA_HOME/agent-tps/runs/`)
- **Windows**: `%APPDATA%/agent-tps/runs/`

---

## 💻 CLI Usage & Commands

You can also use Agent-TPS headlessly or integrate it into CI pipelines:

```bash
# Benchmark an agent with custom prompt
uv run agent-tps bench --provider opencode --prompt "Write a Python binary search"

# Benchmark with strict timeout guards
uv run agent-tps bench --provider claude --ttft-timeout 5.0 --stall-timeout 3.0

# Concurrency stress test: 4 parallel workers, 8 total requests
uv run agent-tps stress --target opencode --concurrency 4 --total 8

# Progressive scalability sweep (1, 2, 4, 8 workers)
uv run agent-tps stress --target codex --sweep --total 16

# View all saved runs in a summary table
uv run agent-tps runs

# Inspect a single report standalone on screen
uv run agent-tps view <RUN_ID>

# Delete a single report or clear all reports
uv run agent-tps delete <RUN_ID>
uv run agent-tps clear --yes

# Inspect past sessions from your local OpenCode SQLite database
uv run agent-tps opencode-history --limit 10
```

---

## 📐 Mathematical Methodology

Agent-TPS computes high-resolution metrics following standardized profiling equations:

$$\text{Decode TPS} = \frac{N_{\text{output}} + N_{\text{reasoning}}}{T_{\text{completed}} - T_{\text{first\_token}}} \times 1000$$

$$\text{End-to-End TPS} = \frac{N_{\text{output}} + N_{\text{reasoning}}}{T_{\text{completed}} - T_{\text{request\_start}}} \times 1000$$

$$\text{Cache Hit Rate} = \frac{N_{\text{cache\_read}}}{N_{\text{input}} + N_{\text{cache\_read}}}$$

$$\text{Throughput Degradation} = \max\left(0, \frac{\text{TPS}_{\text{baseline}} - \text{TPS}_{\text{concurrent}}}{\text{TPS}_{\text{baseline}}}\right) \times 100\%$$

Inter-arrival token jitter is calculated per **RFC 3550**:

$$J = \frac{1}{M-1} \sum_{i=1}^{M-1} |D(i) - D(i-1)|$$

---

## 🛠️ Development & Quality Controls

Agent-TPS enforces strict code quality and formatting powered by **Ruff** (the blazingly fast modern linter/formatter) and **Lefthook**:

```bash
# Run test suite
uv run pytest -v

# Run linter checks
uv run ruff check .

# Format code
uv run ruff format .

# Run pre-commit hooks manually
lefthook run pre-commit
```

---

## 📄 License

MIT © [Sebastian Sala](https://github.com/salasebas)
