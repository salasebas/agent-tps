import pytest

from agent_tps.providers.antigravity import AntigravityRunner
from agent_tps.providers.claude import ClaudeCodeRunner
from agent_tps.providers.codex import CodexRunner
from agent_tps.providers.cursor import CursorRunner
from agent_tps.providers.dispatcher import get_runner_for_provider
from agent_tps.providers.grok import GrokRunner
from agent_tps.providers.opencode import OpenCodeRunner
from agent_tps.providers.opencode_server import OpenCodeServerRunner


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
