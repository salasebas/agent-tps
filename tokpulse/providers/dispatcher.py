from __future__ import annotations

from typing import Any

from tokpulse.providers.antigravity import AntigravityRunner
from tokpulse.providers.base import BaseAgentRunner
from tokpulse.providers.claude import ClaudeCodeRunner
from tokpulse.providers.codex import CodexRunner
from tokpulse.providers.cursor import CursorRunner
from tokpulse.providers.grok import GrokRunner
from tokpulse.providers.opencode import OpenCodeRunner
from tokpulse.providers.opencode_server import OpenCodeServerRunner


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
