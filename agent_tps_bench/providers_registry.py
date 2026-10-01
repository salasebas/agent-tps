from __future__ import annotations

import shutil
import subprocess
from typing import NamedTuple


class ModelOption(NamedTuple):
    provider: str
    model: str
    display: str
    description: str = ""


DEFAULT_MODELS: dict[str, list[dict[str, str]]] = {
    "opencode": [
        {"model": "opencode/longcat-2.5-preview-free", "desc": "Free preview model on OpenCode"},
        {"model": "opencode/big-pickle", "desc": "Large agent model"},
        {"model": "opencode/ling-3.0-flash-fin-free", "desc": "Fast finance/code model (Free)"},
        {"model": "opencode/mimo-v2.6-flash-free", "desc": "Flash speed model (Free)"},
        {"model": "opencode/nemotron-3.5-lightning-free", "desc": "Nvidia Nemotron lightning (Free)"},
        {"model": "opencode-go/glm-5", "desc": "GLM 5 code model via OpenCode Go"},
        {"model": "opencode-go/kimi-k3", "desc": "Moonshot Kimi K3 via OpenCode Go"},
        {"model": "opencode-go/qwen3.8-flash", "desc": "Qwen 3.8 Flash via OpenCode Go"},
        {"model": "openai/gpt-5.4", "desc": "OpenAI GPT-5.4 via OpenCode"},
        {"model": "openai/gpt-5.4-mini", "desc": "OpenAI GPT-5.4 Mini via OpenCode"},
        {"model": "xai/grok-4.20-0309-non-reasoning", "desc": "Grok 4.20 via OpenCode"},
    ],
    "codex": [
        {"model": "gpt-5.4", "desc": "OpenAI Codex next-gen agent"},
        {"model": "gpt-5.4-mini", "desc": "Fast lightweight Codex agent"},
        {"model": "o3", "desc": "OpenAI O3 deep reasoning agent"},
        {"model": "o3-mini", "desc": "OpenAI O3 Mini reasoning"},
        {"model": "o1", "desc": "OpenAI O1 reasoning model"},
        {"model": "gpt-4o", "desc": "OpenAI GPT-4o multimodal model"},
    ],
    "claude": [
        {"model": "claude-3-7-sonnet-20250219", "desc": "Claude 3.7 Sonnet (Hybrid reasoning)"},
        {"model": "claude-3-5-sonnet-20241022", "desc": "Claude 3.5 Sonnet v2 (Best coding)"},
        {"model": "claude-3-5-haiku-20241022", "desc": "Claude 3.5 Haiku (Ultra fast)"},
        {"model": "claude-3-opus-20240229", "desc": "Claude 3 Opus (High complexity)"},
    ],
    "grok": [
        {"model": "grok-3", "desc": "xAI Grok 3 Flagship"},
        {"model": "grok-3-mini", "desc": "xAI Grok 3 Mini fast model"},
        {"model": "grok-2", "desc": "xAI Grok 2"},
        {"model": "grok-build-0.1", "desc": "xAI Grok Build Agent"},
    ],
    "antigravity": [
        {"model": "gemini-2.5-pro", "desc": "Google DeepMind Gemini 2.5 Pro"},
        {"model": "gemini-2.5-flash", "desc": "Google DeepMind Gemini 2.5 Flash"},
        {"model": "gemini-2.0-flash", "desc": "Google Gemini 2.0 Flash"},
        {"model": "gemini-1.5-pro", "desc": "Google Gemini 1.5 Pro (2M context)"},
    ],
    "cursor": [
        {"model": "cursor-fast", "desc": "Cursor Fast model"},
        {"model": "claude-3.5-sonnet", "desc": "Claude 3.5 Sonnet via Cursor"},
        {"model": "gpt-4o", "desc": "GPT-4o via Cursor"},
    ],
    "groq": [
        {"model": "llama-3.3-70b-versatile", "desc": "Llama 3.3 70B on Groq LPU (300-500 TPS)"},
        {"model": "llama-3.1-8b-instant", "desc": "Llama 3.1 8B on Groq LPU (800-1200 TPS)"},
        {"model": "qwen-2.5-coder-32b", "desc": "Qwen 2.5 Coder on Groq LPU"},
        {"model": "deepseek-r1-distill-llama-70b", "desc": "DeepSeek R1 Distill 70B on Groq"},
    ],
    "cerebras": [
        {"model": "llama3.1-70b", "desc": "Llama 3.1 70B on Cerebras CS-3 (1000-1500 TPS)"},
        {"model": "llama3.1-8b", "desc": "Llama 3.1 8B on Cerebras CS-3 (1800-2200 TPS)"},
    ],
    "openrouter": [
        {"model": "anthropic/claude-3.5-sonnet", "desc": "Claude 3.5 Sonnet via OpenRouter"},
        {"model": "deepseek/deepseek-r1", "desc": "DeepSeek R1 Reasoning via OpenRouter"},
        {"model": "deepseek/deepseek-chat", "desc": "DeepSeek V3 via OpenRouter"},
        {"model": "meta-llama/llama-3.3-70b-instruct", "desc": "Llama 3.3 70B via OpenRouter"},
        {"model": "google/gemini-2.0-flash-001", "desc": "Gemini 2.0 Flash via OpenRouter"},
    ],
    "openai": [
        {"model": "gpt-4o", "desc": "GPT-4o Omnimodel"},
        {"model": "gpt-4o-mini", "desc": "GPT-4o Mini lightweight"},
        {"model": "o3-mini", "desc": "OpenAI o3-mini reasoning"},
        {"model": "o1", "desc": "OpenAI o1 full reasoning"},
    ],
    "ollama": [
        {"model": "qwen2.5-coder:7b", "desc": "Qwen 2.5 Coder 7B local"},
        {"model": "llama3.2:3b", "desc": "Llama 3.2 3B local"},
        {"model": "deepseek-r1:8b", "desc": "DeepSeek R1 8B local"},
    ],
}


class ProviderMeta(NamedTuple):
    name: str
    display: str
    kind: str  # "agent" or "api"
    cli_binary: str | None
    is_installed: bool
    description: str


def get_installed_providers() -> list[ProviderMeta]:
    """Inspects the local system for installed agent CLIs and known API providers."""
    providers = [
        ProviderMeta(
            name="opencode",
            display="OpenCode (CLI Agent)",
            kind="agent",
            cli_binary="opencode",
            is_installed=bool(shutil.which("opencode")),
            description="Autonomous coding agent with SQLite session tracking and tools",
        ),
        ProviderMeta(
            name="opencode-server",
            display="OpenCode Server (T3 Code style)",
            kind="agent",
            cli_binary="opencode",
            is_installed=bool(shutil.which("opencode")),
            description="Headless HTTP server mode for zero-lock concurrent subagents",
        ),
        ProviderMeta(
            name="claude",
            display="Claude Code (CLI Agent)",
            kind="agent",
            cli_binary="claude",
            is_installed=bool(shutil.which("claude")),
            description="Anthropic Claude Code CLI with stream-json event telemetry",
        ),
        ProviderMeta(
            name="codex",
            display="Codex (CLI Agent)",
            kind="agent",
            cli_binary="codex",
            is_installed=bool(shutil.which("codex")),
            description="OpenAI Codex CLI for autonomous software development",
        ),
        ProviderMeta(
            name="grok",
            display="Grok Build (CLI Agent)",
            kind="agent",
            cli_binary="grok",
            is_installed=bool(shutil.which("grok")),
            description="xAI Grok Build terminal agent",
        ),
        ProviderMeta(
            name="cursor",
            display="Cursor CLI (Agent)",
            kind="agent",
            cli_binary="cursor",
            is_installed=bool(shutil.which("cursor")),
            description="Cursor CLI interface for AI pair programming",
        ),
        ProviderMeta(
            name="antigravity",
            display="Google Antigravity (Agent)",
            kind="agent",
            cli_binary="agy",
            is_installed=bool(shutil.which("agy")) or True,
            description="Google DeepMind Antigravity agent harness and Gemini models",
        ),
        ProviderMeta(
            name="groq",
            display="Groq (Ultra-Fast LPU API)",
            kind="api",
            cli_binary=None,
            is_installed=True,
            description="Groq hardware inference (300-1500 TPS)",
        ),
        ProviderMeta(
            name="cerebras",
            display="Cerebras (Wafer-Scale CS-3 API)",
            kind="api",
            cli_binary=None,
            is_installed=True,
            description="Cerebras wafer-scale engine (1000-2200 TPS)",
        ),
        ProviderMeta(
            name="openrouter",
            display="OpenRouter (Multi-Model Router API)",
            kind="api",
            cli_binary=None,
            is_installed=True,
            description="Unified gateway to 100+ models with latency routing",
        ),
        ProviderMeta(
            name="openai",
            display="OpenAI Direct (API)",
            kind="api",
            cli_binary=None,
            is_installed=True,
            description="Direct OpenAI platform chat/completions endpoint",
        ),
        ProviderMeta(
            name="ollama",
            display="Ollama Local (API)",
            kind="api",
            cli_binary="ollama",
            is_installed=bool(shutil.which("ollama")),
            description="Self-hosted local models on your GPU or Apple Silicon",
        ),
    ]
    return providers


def get_models_for_provider(provider_name: str) -> list[ModelOption]:
    """Returns available models for a given provider, querying dynamic CLI models if available."""
    normalized = provider_name.lower().replace("-server", "")

    # For OpenCode, attempt dynamic discovery
    if normalized == "opencode" and shutil.which("opencode"):
        try:
            out = subprocess.check_output(["opencode", "models"], text=True, timeout=2.0)
            lines = [line.strip() for line in out.splitlines() if line.strip()]
            if lines:
                return [
                    ModelOption(
                        provider=provider_name,
                        model=m,
                        display=f"{m}",
                        description="Live model from OpenCode catalog",
                    )
                    for m in lines[:30]
                ]
        except Exception:
            pass

    static_list = DEFAULT_MODELS.get(normalized, [])
    return [
        ModelOption(
            provider=provider_name,
            model=item["model"],
            display=item["model"],
            description=item["desc"],
        )
        for item in static_list
    ]


def get_all_models_flat() -> list[ModelOption]:
    """Returns all models across all providers for global search."""
    all_models: list[ModelOption] = []
    providers = get_installed_providers()

    for p in providers:
        models = get_models_for_provider(p.name)
        for m in models:
            all_models.append(
                ModelOption(
                    provider=p.name,
                    model=m.model,
                    display=f"[{p.name.upper()}] {m.model}",
                    description=f"{p.display} • {m.description}",
                )
            )
    return all_models
