"""UI module for Agent-TPS."""

from agent_tps.ui.interactive import run_interactive_tui
from agent_tps.ui.reporter import (
    render_benchmark_result,
    render_concurrency_report,
    render_opencode_sessions,
    render_runs_table,
    render_saved_run_detail,
)

__all__ = [
    "render_benchmark_result",
    "render_concurrency_report",
    "render_opencode_sessions",
    "render_runs_table",
    "render_saved_run_detail",
    "run_interactive_tui",
]
