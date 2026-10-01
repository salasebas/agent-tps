from __future__ import annotations

from typing import Any

from agent_tps.providers.antigravity import AntigravityRunner
from agent_tps.providers.base import BaseAgentRunner
from agent_tps.providers.claude import ClaudeCodeRunner
from agent_tps.providers.codex import CodexRunner
from agent_tps.providers.cursor import CursorRunner
from agent_tps.providers.grok import GrokRunner
from agent_tps.providers.opencode import OpenCodeRunner
from agent_tps.providers.opencode_server import OpenCodeServerRunner


def get_runner_for_provider(provider: str, **kwargs: Any) -> BaseAgentRunner:
    """Instantiates and returns the corresponding agent runner for a given provider name."""
    p_lower = provider.lower().strip()

    if p_lower in ("opencode", "opencode-cli"):
        return OpenCodeRunner(
            binary_path=kwargs.get("binary_path", "opencode"),
            auto_cleanup=kwargs.get("auto_cleanup", True),
        )

    elif p_lower in ("opencode-server", "opencode-srv"):
        return OpenCodeServerRunner(
            base_url=kwargs.get("base_url", "http://127.0.0.1:4096"),
            auto_spawn=kwargs.get("auto_spawn", True),
            auto_cleanup=kwargs.get("auto_cleanup", True),
        )

    elif p_lower in ("cursor", "cursor-agent"):
        return CursorRunner(
            binary_path=kwargs.get("binary_path"),
            auto_cleanup=kwargs.get("auto_cleanup", True),
        )

    elif p_lower in ("grok", "grok-build"):
        return GrokRunner(
            binary_path=kwargs.get("binary_path"),
            auto_cleanup=kwargs.get("auto_cleanup", True),
        )

    elif p_lower in ("antigravity", "agy"):
        return AntigravityRunner(
            binary_path=kwargs.get("binary_path"),
            auto_cleanup=kwargs.get("auto_cleanup", True),
        )

    elif p_lower in ("codex", "codex-cli"):
        return CodexRunner(
            binary_path=kwargs.get("binary_path", "codex"),
            auto_cleanup=kwargs.get("auto_cleanup", True),
        )

    elif p_lower in ("claude", "claude-code"):
        return ClaudeCodeRunner(
            binary_path=kwargs.get("binary_path", "claude"),
            auto_cleanup=kwargs.get("auto_cleanup", True),
        )

    else:
        # Default fallback to OpenCode
        return OpenCodeRunner()
