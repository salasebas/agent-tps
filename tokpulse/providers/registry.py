"""Provider & Model registry for TokPulse.

Includes only the coding agent CLI engines supported by T3 Code:
- OpenCode (opencode)
- Cursor (cursor-agent / cursor)
- Grok (grok / grok-build)
- Antigravity (agy / antigravity)
- Codex (codex)
- Claude Code (claude)
"""

from __future__ import annotations

import shutil
from typing import Any

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    id: str
    name: str
    description: str = ""
    is_default: bool = False


class ProviderInfo(BaseModel):
    id: str
    name: str
    binary: str
    description: str
    default_model: str
    models: list[ModelInfo] = Field(default_factory=list)

    @property
    def is_installed(self) -> bool:
        return bool(shutil.which(self.binary))


PROVIDERS_CATALOG: dict[str, ProviderInfo] = {
    "opencode": ProviderInfo(
        id="opencode",
        name="OpenCode",
        binary="opencode",
        description="Local agentic coding assistant engine with SQLite telemetry",
        default_model="opencode/longcat-2.5-preview-free",
        models=[
            ModelInfo(
                id="opencode/longcat-2.5-preview-free",
                name="LongCat 2.5 Preview (Free)",
                description="Built-in free tier agent model",
                is_default=True,
            ),
            ModelInfo(
                id="openai/gpt-5",
                name="GPT-5 (OpenCode)",
                description="High capability reasoning model",
            ),
            ModelInfo(
                id="openai/gpt-5.4",
                name="GPT-5.4 (OpenCode)",
                description="Ultra fast coding model",
            ),
            ModelInfo(
                id="anthropic/claude-3-7-sonnet",
                name="Claude 3.7 Sonnet (OpenCode)",
                description="Hybrid thinking & fast coding agent",
            ),
            ModelInfo(
                id="deepseek/deepseek-chat",
                name="DeepSeek V3 (OpenCode)",
                description="Cost-effective high throughput model",
            ),
        ],
    ),
    "cursor": ProviderInfo(
        id="cursor",
        name="Cursor Agent",
        binary="cursor-agent",
        description="Cursor's agentic CLI runtime (cursor-agent / cursor)",
        default_model="auto",
        models=[
            ModelInfo(
                id="auto", name="Auto (Cursor default)", description="Cursor smart router", is_default=True
            ),
            ModelInfo(id="composer-2", name="Composer 2", description="Cursor specialized composer model"),
            ModelInfo(id="composer-1.5", name="Composer 1.5", description="Stable composer runtime"),
            ModelInfo(
                id="claude-opus-4-6", name="Claude Opus 4.6", description="Opus thinking engine in Cursor"
            ),
            ModelInfo(id="claude-sonnet-4-6", name="Claude Sonnet 4.6", description="Balanced agentic model"),
        ],
    ),
    "grok": ProviderInfo(
        id="grok",
        name="Grok Agent",
        binary="grok-build",
        description="xAI Grok terminal coding agent (grok / grok-build)",
        default_model="grok-build",
        models=[
            ModelInfo(
                id="grok-build",
                name="Grok Build (Current)",
                description="Active coding agent session model",
                is_default=True,
            ),
            ModelInfo(id="grok-3", name="Grok 3", description="Flagship reasoning and code intelligence"),
            ModelInfo(id="grok-2", name="Grok 2", description="High-speed assistant model"),
        ],
    ),
    "antigravity": ProviderInfo(
        id="antigravity",
        name="Antigravity",
        binary="agy",
        description="Google Antigravity agent CLI runtime (agy)",
        default_model="antigravity-default",
        models=[
            ModelInfo(
                id="antigravity-default",
                name="Antigravity Session Default",
                description="Maintains active IDE/CLI session model",
                is_default=True,
            ),
            ModelInfo(
                id="gemini-2.5-pro", name="Gemini 2.5 Pro", description="Deep multi-step reasoning agent"
            ),
            ModelInfo(
                id="gemini-2.5-flash", name="Gemini 2.5 Flash", description="Sub-second low latency agent"
            ),
        ],
    ),
    "codex": ProviderInfo(
        id="codex",
        name="Codex",
        binary="codex",
        description="OpenAI Codex CLI execution runtime (codex exec)",
        default_model="gpt-6-astra",
        models=[
            ModelInfo(
                id="gpt-6-astra",
                name="GPT-6 Astra",
                description="Primary Codex default model",
                is_default=True,
            ),
            ModelInfo(id="gpt-5.6-sol", name="GPT-5.6 Sol", description="Fast low-latency agent model"),
            ModelInfo(
                id="gpt-5.6-terra", name="GPT-5.6 Terra", description="Robust coding and refactoring model"
            ),
            ModelInfo(id="gpt-5.4", name="GPT-5.4", description="Modern stable coding architecture"),
            ModelInfo(
                id="gpt-5.3-codex", name="GPT-5.3 Codex", description="Established developer agent engine"
            ),
        ],
    ),
    "claude": ProviderInfo(
        id="claude",
        name="Claude Code",
        binary="claude",
        description="Anthropic Claude Code CLI agent (claude -p)",
        default_model="claude-fable-5-1",
        models=[
            ModelInfo(
                id="claude-fable-5-1",
                name="Claude Fable 5.1",
                description="T3 Code primary Claude driver model",
                is_default=True,
            ),
            ModelInfo(
                id="claude-haiku-4-5", name="Claude Haiku 4.5", description="Fast text generation default"
            ),
            ModelInfo(
                id="claude-3-7-sonnet-latest",
                name="Claude 3.7 Sonnet",
                description="Hybrid thinking & execution agent",
            ),
            ModelInfo(
                id="claude-3-5-sonnet-latest",
                name="Claude 3.5 Sonnet",
                description="High precision coding model",
            ),
        ],
    ),
}


def detect_available_providers() -> list[ProviderInfo]:
    """Returns list of providers whose CLI binary is found on PATH."""
    available: list[ProviderInfo] = []
    for prov in PROVIDERS_CATALOG.values():
        has_bin = bool(shutil.which(prov.binary))
        if not has_bin and (
            (prov.id == "cursor" and shutil.which("cursor"))
            or (prov.id == "grok" and shutil.which("grok"))
            or (prov.id == "antigravity" and shutil.which("antigravity"))
        ):
            has_bin = True
        if has_bin:
            available.append(prov)
    return available


def get_all_models_flat() -> list[dict[str, Any]]:
    """Returns a flat list of models across all providers for instant fuzzy search."""
    flat: list[dict[str, Any]] = []
    for prov_id, prov in PROVIDERS_CATALOG.items():
        is_inst = prov.is_installed
        inst_label = "Installed" if is_inst else "Not in PATH"
        for m in prov.models:
            flat.append(
                {
                    "provider": prov_id,
                    "provider_name": prov.name,
                    "model_id": m.id,
                    "model_name": m.name,
                    "description": m.description,
                    "installed": is_inst,
                    "display": f"{prov.name} · {m.name} [{m.id}] ({inst_label})",
                }
            )
    return flat
