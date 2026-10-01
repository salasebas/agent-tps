from pathlib import Path

import pytest

from agent_tps.core.models import BenchmarkStatus, TimeoutType
from agent_tps.providers.opencode import OpenCodeRunner


@pytest.fixture
def fake_opencode_script(tmp_path: Path) -> Path:
    """Creates a mock opencode executable script that outputs json events."""
    script_path = tmp_path / "mock_opencode"
    code = """#!/usr/bin/env python3
import sys, time, json

if "--format" in sys.argv and "json" in sys.argv:
    # 1. step_start
    print(json.dumps({"type": "step_start", "timestamp": int(time.time()*1000), "sessionID": "ses_mock_exec"}))
    sys.stdout.flush()
    time.sleep(0.05)

    # 2. text tokens
    print(json.dumps({
        "type": "text",
        "timestamp": int(time.time()*1000),
        "sessionID": "ses_mock_exec",
        "part": {"type": "text", "text": "Mock response line 1\\n"}
    }))
    sys.stdout.flush()
    time.sleep(0.05)

    print(json.dumps({
        "type": "text",
        "timestamp": int(time.time()*1000),
        "sessionID": "ses_mock_exec",
        "part": {"type": "text", "text": "Mock response line 2\\n"}
    }))
    sys.stdout.flush()

    # 3. step_finish
    print(json.dumps({
        "type": "step_finish",
        "timestamp": int(time.time()*1000),
        "sessionID": "ses_mock_exec",
        "part": {
            "type": "step-finish",
            "tokens": {"input": 50, "output": 25, "reasoning": 10, "cache": {"read": 100, "write": 0}}
        }
    }))
    sys.stdout.flush()
"""
    script_path.write_text(code)
    script_path.chmod(0o755)
    return script_path


@pytest.mark.asyncio
async def test_opencode_runner_success(fake_opencode_script: Path):
    runner = OpenCodeRunner(binary_path=str(fake_opencode_script))
    chunks = []
    res = await runner.run_prompt(
        prompt="Test",
        on_chunk=lambda c: chunks.append(c),
    )

    assert res.status == BenchmarkStatus.SUCCESS
    assert res.timeout_type == TimeoutType.NONE
    assert res.id == "ses_mock_exec"
    assert res.tokens.input_tokens == 50
    assert res.tokens.output_tokens == 25
    assert res.tokens.reasoning_tokens == 10
    assert res.tokens.cached_read_tokens == 100
    assert len(chunks) == 2
    assert res.tps.decode_tps > 0.0
    assert res.timings.ttft_ms is not None


@pytest.mark.asyncio
async def test_opencode_runner_ttft_timeout(tmp_path: Path):
    script_path = tmp_path / "mock_slow_opencode"
    code = """#!/usr/bin/env python3
import time
time.sleep(1.0)
"""
    script_path.write_text(code)
    script_path.chmod(0o755)

    runner = OpenCodeRunner(binary_path=str(script_path))
    res = await runner.run_prompt(
        prompt="Test",
        ttft_timeout_s=0.1,
    )

    assert res.status == BenchmarkStatus.TIMEOUT
    assert res.timeout_type == TimeoutType.TTFT_TIMEOUT
