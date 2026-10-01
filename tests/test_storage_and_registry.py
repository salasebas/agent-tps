from pathlib import Path
from agent_tps_bench.models import (
    BenchmarkResult,
    BenchmarkStatus,
    TimeoutType,
    TimingMetrics,
    TokenMetrics,
    TPSMetrics,
)
from agent_tps_bench.providers_registry import (
    get_all_models_flat,
    get_installed_providers,
    get_models_for_provider,
)
from agent_tps_bench.storage import BenchmarkStorage


def test_storage_save_and_list(tmp_path: Path):
    storage = BenchmarkStorage(storage_dir=tmp_path)
    res = BenchmarkResult(
        id="test-run-123",
        provider="opencode",
        model="longcat-2.5",
        status=BenchmarkStatus.SUCCESS,
        timeout_type=TimeoutType.NONE,
        tokens=TokenMetrics(input_tokens=100, output_tokens=50),
        timings=TimingMetrics(request_start_ms=0, completed_ms=1000, ttft_ms=200),
        tps=TPSMetrics(decode_tps=50.0, e2e_tps=50.0),
        raw_response_preview="Hello world",
    )
    saved_path = storage.save_run(res, prompt="Test prompt", full_output="Full output text")
    assert saved_path.exists()

    runs = storage.list_runs()
    assert len(runs) == 1
    assert runs[0]["provider"] == "opencode"
    assert runs[0]["model"] == "longcat-2.5"
    assert runs[0]["decode_tps"] == 50.0

    loaded = storage.load_run("test-run-123")
    assert loaded is not None
    assert loaded["prompt"] == "Test prompt"
    assert loaded["full_output"] == "Full output text"


def test_providers_registry():
    providers = get_installed_providers()
    names = [p.name for p in providers]
    assert "opencode" in names
    assert "claude" in names
    assert "codex" in names
    assert "grok" in names
    assert "cursor" in names
    assert "antigravity" in names
    assert "groq" in names

    # OpenCode models
    models = get_models_for_provider("opencode")
    assert len(models) > 0

    # Claude models
    claude_models = get_models_for_provider("claude")
    assert any("sonnet" in m.model for m in claude_models)

    # Flat global search list
    flat = get_all_models_flat()
    assert len(flat) > 20
