from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agent_tps_bench.models import (
    BenchmarkResult,
    BenchmarkStatus,
    ConcurrencyReport,
    TimeoutType,
)
from agent_tps_bench.opencode_reader import OpenCodeSessionDetail


console = Console()


def render_benchmark_result(result: BenchmarkResult, console: Console = console) -> None:
    """Renders a single benchmark result with rich styling."""
    status_style = "green bold" if result.status == BenchmarkStatus.SUCCESS else "red bold"
    if result.status == BenchmarkStatus.RATE_LIMITED:
        status_style = "yellow bold"
    elif result.status == BenchmarkStatus.TIMEOUT:
        status_style = "magenta bold"

    table = Table(title=f"Benchmark Result: {result.provider} / {result.model}", show_header=True)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="bold")
    table.add_column("Description", style="dim")

    table.add_row("Status", Text(result.status.value.upper(), style=status_style), "Overall outcome")
    table.add_row(
        "Timeout / Error Class",
        Text(result.timeout_type.value, style="yellow" if result.timeout_type != TimeoutType.NONE else "green"),
        "Specific classification if stalled or timed out",
    )

    # TPS
    table.add_row(
        "Decode TPS",
        f"[green bold]{result.tps.decode_tps:.2f} tok/s[/green bold]",
        "Active generation throughput (tokens / gen time)",
    )
    table.add_row(
        "End-to-End TPS",
        f"[cyan]{result.tps.e2e_tps:.2f} tok/s[/cyan]",
        "Wall-clock speed including TTFT / prefill",
    )
    table.add_row(
        "Total Throughput",
        f"{result.tps.total_throughput_tps:.2f} tok/s",
        "All tokens (input + output + reasoning) / total duration",
    )

    # Latencies
    ttft_str = f"{result.timings.ttft_ms:.1f} ms" if result.timings.ttft_ms is not None else "N/A"
    table.add_row("Time to First Token (TTFT)", f"[yellow bold]{ttft_str}[/yellow bold]", "Prompt eval + network latency")

    gen_dur_str = f"{result.timings.generation_duration_ms:.1f} ms" if result.timings.generation_duration_ms else "N/A"
    table.add_row("Generation Duration", gen_dur_str, "Time spent emitting tokens")

    tot_dur_str = f"{result.timings.total_duration_ms:.1f} ms" if result.timings.total_duration_ms else "N/A"
    table.add_row("Total Request Duration", tot_dur_str, "Total time from request initiation to completion")

    # Inter-token latency
    if result.timings.inter_token_latencies_ms:
        table.add_row("ITL p50 / p90 / p99", f"{result.timings.itl_p50_ms:.1f} / {result.timings.itl_p90_ms:.1f} / {result.timings.itl_p99_ms:.1f} ms", "Median and tail token gap")
        table.add_row("Max ITL", f"{result.timings.itl_max_ms:.1f} ms", "Longest stall between tokens")
        table.add_row("Jitter", f"{result.timings.jitter_ms:.1f} ms", "Arrival variance (smoothness)")

    # Tokens
    tok = result.tokens
    table.add_row("Output Tokens", f"[green]{tok.output_tokens}[/green]", "Completion tokens generated")
    table.add_row("Reasoning Tokens", f"[magenta]{tok.reasoning_tokens}[/magenta]", "Internal thinking tokens")
    table.add_row("Input Tokens", f"{tok.input_tokens}", "Prompt tokens evaluated")
    table.add_row("Cache Hit Rate", f"{tok.cache_hit_rate * 100:.1f}% ({tok.cached_read_tokens} tokens)", "Prompt cache savings")

    if result.error_message:
        table.add_row("Error Detail", f"[red]{result.error_message}[/red]", "Error or timeout explanation")
    if result.retry_after_s:
        table.add_row("Retry-After", f"[yellow]{result.retry_after_s} s[/yellow]", "Rate limit backoff duration")

    console.print(table)


def render_opencode_sessions(sessions: list[OpenCodeSessionDetail], console: Console = console) -> None:
    """Renders a summary table of recent OpenCode sessions."""
    table = Table(title="OpenCode Sessions History & TPS Telemetry", show_header=True)
    table.add_column("Session ID", style="dim", no_wrap=True)
    table.add_column("Title", style="bold")
    table.add_column("Model", style="cyan")
    table.add_column("Tokens (In/Out/Reas)", style="magenta")
    table.add_column("Cache Hit", style="blue")
    table.add_column("Gen Dur", style="yellow")
    table.add_column("Decode TPS", style="green bold")
    table.add_column("E2E TPS", style="cyan")
    table.add_column("Cost", style="dim")

    for s in sessions:
        tok = s.tokens
        tokens_str = f"{tok.input_tokens} / {tok.output_tokens} / {tok.reasoning_tokens}"
        cache_str = f"{tok.cache_hit_rate * 100:.0f}%"
        gen_dur = f"{s.timings.generation_duration_ms / 1000.0:.2f}s" if s.timings.generation_duration_ms else f"{s.duration_ms / 1000.0:.2f}s"
        cost_str = f"${s.cost:.4f}" if s.cost > 0 else "$0.00"

        table.add_row(
            s.session_id[:16] + "…",
            s.title[:24],
            s.model[:20],
            tokens_str,
            cache_str,
            gen_dur,
            f"{s.tps.decode_tps:.1f} tps",
            f"{s.tps.e2e_tps:.1f} tps",
            cost_str,
        )

    console.print(table)


def render_concurrency_report(report: ConcurrencyReport, console: Console = console) -> None:
    """Renders a concurrency batch report."""
    summary_panel = Panel(
        f"[bold]Concurrency Level:[/bold] {report.concurrency_level} workers\n"
        f"[bold]Total Requests:[/bold] {report.total_requests} "
        f"([green]{report.successful_requests} OK[/green], "
        f"[yellow]{report.rate_limited_requests} 429 RateLimit[/yellow], "
        f"[magenta]{report.timed_out_requests} Timed Out[/magenta], "
        f"[red]{report.failed_requests} Error[/red])\n"
        f"[bold]Wall Clock Duration:[/bold] {report.wall_clock_duration_s:.2f}s\n"
        f"[bold green]Aggregate Decode TPS:[/bold green] [bold green]{report.aggregate_decode_tps:.2f} tok/s[/bold green]\n"
        f"[bold cyan]Aggregate E2E TPS:[/bold cyan] {report.aggregate_e2e_tps:.2f} tok/s\n"
        f"[bold]Mean Worker Decode TPS:[/bold] {report.mean_worker_decode_tps:.2f} tok/s\n"
        f"[bold yellow]Concurrency Degradation:[/bold yellow] {report.degradation_percent:.1f}%\n"
        f"[bold]TTFT Latency:[/bold] Mean={report.mean_ttft_ms:.1f}ms | p50={report.p50_ttft_ms:.1f}ms | p95={report.p95_ttft_ms:.1f}ms | p99={report.p99_ttft_ms:.1f}ms",
        title="[bold yellow]Subagents / Concurrency Stress Report[/bold yellow]",
        border_style="cyan",
    )
    console.print(summary_panel)


def render_concurrency_sweep(reports: list[ConcurrencyReport], console: Console = console) -> None:
    """Renders a comparison table across multiple concurrency levels."""
    table = Table(title="Concurrency Scaling & Rate-Limit Degradation Curve", show_header=True)
    table.add_column("Workers", style="cyan bold")
    table.add_column("Success / Total", style="bold")
    table.add_column("Agg Decode TPS", style="green bold")
    table.add_column("Worker TPS", style="green")
    table.add_column("Degradation", style="yellow")
    table.add_column("p50 TTFT", style="blue")
    table.add_column("p95 TTFT", style="blue")
    table.add_column("429 RateLimits", style="red")

    for r in reports:
        deg_style = "green" if r.degradation_percent < 15 else "yellow" if r.degradation_percent < 40 else "red bold"
        table.add_row(
            str(r.concurrency_level),
            f"{r.successful_requests}/{r.total_requests}",
            f"{r.aggregate_decode_tps:.1f} tok/s",
            f"{r.mean_worker_decode_tps:.1f} tok/s",
            Text(f"{r.degradation_percent:.1f}%", style=deg_style),
            f"{r.p50_ttft_ms:.1f} ms",
            f"{r.p95_ttft_ms:.1f} ms",
            Text(str(r.rate_limited_requests), style="red bold" if r.rate_limited_requests > 0 else "dim"),
        )

    console.print(table)


def export_report_to_json(data: Any, output_path: str | Path) -> None:
    """Exports benchmark results to a formatted JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dumpable = data.model_dump() if hasattr(data, "model_dump") else [d.model_dump() if hasattr(d, "model_dump") else d for d in data] if isinstance(data, list) else data
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dumpable, f, indent=2)
