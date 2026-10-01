from pathlib import Path

from agent_tps.core.models import (
    BenchmarkResult,
    BenchmarkStatus,
    TimeoutType,
    TimingMetrics,
    TokenMetrics,
    TPSMetrics,
)
from agent_tps.providers.registry import PROVIDERS_CATALOG, get_all_models_flat
from agent_tps.storage.store import BenchmarkStorage


def test_storage_save_list_and_privacy(tmp_path: Path):
    storage = BenchmarkStorage(runs_dir=tmp_path)
    res = BenchmarkResult(
        id="test-run-123",
        provider="opencode",
        model="longcat-2.5",
        status=BenchmarkStatus.SUCCESS,
        timeout_type=TimeoutType.NONE,
        tokens=TokenMetrics(input_tokens=100, output_tokens=50),
        timings=TimingMetrics(request_start_ms=0, completed_ms=1000, ttft_ms=200),
        tps=TPSMetrics(decode_tps=50.0, e2e_tps=50.0),
    )
    saved_path = storage.save_run(res)
    assert saved_path.exists()

    runs = storage.list_runs()
    assert len(runs) == 1
    assert runs[0]["provider"] == "opencode"
    assert runs[0]["model"] == "longcat-2.5"
    assert runs[0]["tps"]["decode_tps"] == 50.0

    # Privacy verification: no raw prompt or chat output stored
    assert "prompt" not in runs[0]
    assert "full_output" not in runs[0]

    # Get single run
    loaded = storage.get_run("test-run-123")
    assert loaded is not None
    assert loaded["id"] == "test-run-123"

    # Delete single run
    deleted = storage.delete_run("test-run-123")
    assert deleted is True
    assert len(storage.list_runs()) == 0


def test_storage_clear_all(tmp_path: Path):
    storage = BenchmarkStorage(runs_dir=tmp_path)
    for i in range(3):
        res = BenchmarkResult(
            id=f"run-{i}",
            provider="opencode",
            model="default",
            status=BenchmarkStatus.SUCCESS,
            timings=TimingMetrics(request_start_ms=0),
        )
        storage.save_run(res)

    assert len(storage.list_runs()) == 3
    cleared = storage.clear_all_runs()
    assert cleared == 3
    assert len(storage.list_runs()) == 0


def test_storage_save_concurrency_run(tmp_path: Path):
    from agent_tps.core.calculator import compute_concurrency_report

    storage = BenchmarkStorage(runs_dir=tmp_path)
    res_fail = BenchmarkResult(
        id="worker-1",
        provider="antigravity",
        model="default",
        status=BenchmarkStatus.ERROR,
        timeout_type=TimeoutType.PROCESS_CRASH,
        error_message="Binary 'agy' not found in PATH.",
        timings=TimingMetrics(request_start_ms=0),
    )
    report = compute_concurrency_report(
        concurrency_level=4,
        results=[res_fail, res_fail, res_fail, res_fail],
        wall_clock_duration_s=0.05,
    )

    saved_path = storage.save_concurrency_run(report, target="antigravity")
    assert saved_path.exists()

    runs = storage.list_runs()
    assert len(runs) == 1
    assert runs[0]["type"] == "stress"
    assert runs[0]["provider"] == "antigravity"
    assert runs[0]["status"] == "FAILED"
    assert runs[0]["failed_requests"] == 4
    assert len(runs[0]["sample_errors"]) == 1
    assert "not found in PATH" in runs[0]["sample_errors"][0]


def test_providers_registry_t3code_drivers():
    # Only T3 Code coding agent drivers must be present
    expected_providers = {"opencode", "cursor", "grok", "antigravity", "codex", "claude"}
    assert set(PROVIDERS_CATALOG.keys()) == expected_providers

    # Ensure groq and openrouter are NOT present
    assert "groq" not in PROVIDERS_CATALOG
    assert "openrouter" not in PROVIDERS_CATALOG

    # Models catalog checks
    assert len(PROVIDERS_CATALOG["opencode"].models) > 0
    assert any("fable" in m.id for m in PROVIDERS_CATALOG["claude"].models)
    assert any("astra" in m.id for m in PROVIDERS_CATALOG["codex"].models)
    assert any("grok" in m.id for m in PROVIDERS_CATALOG["grok"].models)
    assert any("composer" in m.id for m in PROVIDERS_CATALOG["cursor"].models)
    assert any("antigravity" in m.id for m in PROVIDERS_CATALOG["antigravity"].models)

    # Flat fuzzy search list
    flat = get_all_models_flat()
    assert len(flat) >= 15
