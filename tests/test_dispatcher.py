import pytest

from tokpulse.providers.antigravity import AntigravityRunner
from tokpulse.providers.claude import ClaudeCodeRunner
from tokpulse.providers.codex import CodexRunner
from tokpulse.providers.cursor import CursorRunner
from tokpulse.providers.dispatcher import get_runner_for_provider
from tokpulse.providers.grok import GrokRunner
from tokpulse.providers.opencode import OpenCodeRunner
from tokpulse.providers.opencode_server import OpenCodeServerRunner


def test_dispatcher_resolution():
    assert isinstance(get_runner_for_provider("opencode"), OpenCodeRunner)
    assert isinstance(get_runner_for_provider("opencode-server"), OpenCodeServerRunner)
    assert isinstance(get_runner_for_provider("cursor"), CursorRunner)
    assert isinstance(get_runner_for_provider("grok"), GrokRunner)
    assert isinstance(get_runner_for_provider("antigravity"), AntigravityRunner)
    assert isinstance(get_runner_for_provider("codex"), CodexRunner)
    assert isinstance(get_runner_for_provider("claude"), ClaudeCodeRunner)


@pytest.mark.asyncio
async def test_agent_missing_binary_graceful_error():
    # Runners should return a graceful PROCESS_CRASH / ERROR if binary isn't in PATH
    runner = ClaudeCodeRunner(binary_path="/nonexistent/binary/path")
    res = await runner.run_prompt("Test prompt")
    assert res.status.value == "error"
    assert "not found" in res.error_message.lower()
