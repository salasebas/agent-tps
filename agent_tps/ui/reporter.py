"""Rich terminal reporting components and cards for Agent-TPS."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agent_tps.core.models import BenchmarkResult, BenchmarkStatus, ConcurrencyReport, TimeoutType
from agent_tps.providers.opencode import OpenCodeSessionDetail

console = Console()


def render_benchmark_result(result: BenchmarkResult) -> None:
    """Renders a standalone result card for a benchmark execution."""
    status_style = {
        BenchmarkStatus.SUCCESS: "bold green",
        BenchmarkStatus.TIMEOUT: "bold yellow",
        BenchmarkStatus.RATE_LIMITED: "bold magenta",
        BenchmarkStatus.ERROR: "bold red",
    }.get(result.status, "bold white")

    title_text = Text()
    title_text.append("⚡ Agent-TPS Result: ", style="bold cyan")
    title_text.append(f"{result.provider.upper()} ", style="bold white")
    title_text.append(f"({result.model})", style="dim")

    summary_table = Table.grid(padding=(0, 2))
    summary_table.add_column("Key", style="dim")
    summary_table.add_column("Value", style="bold")

    summary_table.add_row("Status", f"[{status_style}]{result.status.value.upper()}[/{status_style}]")
    if result.timeout_type != TimeoutType.NONE:
        summary_table.add_row("Failure Classification", f"[red]{result.timeout_type.value}[/red]")
    if result.error_message:
        summary_table.add_row("Details", f"[red]{result.error_message[:120]}[/red]")

    # Metrics table
    metrics_table = Table(title="Performance Telemetry", show_header=True, header_style="bold cyan")
    metrics_table.add_column("Metric", style="dim")
    metrics_table.add_column("Value", justify="right", style="bold yellow")
    metrics_table.add_column("Target / Notes", style="dim")

    # TPS
    metrics_table.add_row(
        "Decode TPS",
        f"{result.tps.decode_tps:.2f} tok/s",
        "Raw generation speed (excl. TTFT)",
    )
    metrics_table.add_row(
        "End-to-End TPS",
        f"{result.tps.e2e_tps:.2f} tok/s",
        "Perceived user velocity (turn start -> finish)",
    )
    if result.tps.total_throughput_tps > 0:
        metrics_table.add_row(
            "Total Throughput",
            f"{result.tps.total_throughput_tps:.2f} tok/s",
            "Total input + output processed / second",
        )

    # Latencies
    if result.timings.ttft_ms is not None:
        ttft_color = (
            "green" if result.timings.ttft_ms < 1500 else "yellow" if result.timings.ttft_ms < 4000 else "red"
        )
        metrics_table.add_row(
            "Time to First Token (TTFT)",
            f"[{ttft_color}]{result.timings.ttft_ms:.1f} ms[/{ttft_color}]",
            "Initial latency until first token",
        )

    if result.timings.generation_duration_ms is not None:
        metrics_table.add_row(
            "Generation Duration",
            f"{result.timings.generation_duration_ms:.1f} ms",
            "Active token streaming window",
        )

    if result.timings.total_duration_ms is not None:
        metrics_table.add_row(
            "Total Request Duration",
            f"{result.timings.total_duration_ms:.1f} ms",
            "Full round-trip wall-clock duration",
        )

    # ITL Percentiles
    if result.timings.inter_token_latencies_ms:
        metrics_table.add_row(
            "ITL p50 / p90 / p99",
            f"{result.timings.itl_p50_ms:.1f} / {result.timings.itl_p90_ms:.1f} / {result.timings.itl_p99_ms:.1f} ms",
            "Inter-token delay percentiles",
        )
        metrics_table.add_row(
            "Jitter (RFC 3550)",
            f"{result.timings.jitter_ms:.2f} ms",
            "Inter-token arrival variance",
        )

    # Tokens
    tokens_table = Table(title="Token Accounting", show_header=True, header_style="bold magenta")
    tokens_table.add_column("Type", style="dim")
    tokens_table.add_column("Count", justify="right", style="bold")

    tokens_table.add_row("Input Tokens", str(result.tokens.input_tokens))
    tokens_table.add_row("Output Tokens", str(result.tokens.output_tokens))
    if result.tokens.reasoning_tokens > 0:
        tokens_table.add_row("Reasoning Tokens", str(result.tokens.reasoning_tokens))
    if result.tokens.cached_read_tokens > 0:
        tokens_table.add_row("Cache Read", str(result.tokens.cached_read_tokens))
    if result.tokens.cached_write_tokens > 0:
        tokens_table.add_row("Cache Write", str(result.tokens.cached_write_tokens))
    tokens_table.add_row("Generated Tokens (Total)", str(result.tokens.generated_tokens))
    tokens_table.add_row("Cache Hit Rate", f"{result.tokens.cache_hit_rate * 100:.1f}%")

    main_grid = Table.grid(padding=(1, 2))
    main_grid.add_column()
    main_grid.add_row(summary_table)
    main_grid.add_row(metrics_table)
    main_grid.add_row(tokens_table)

    console.print()
    console.print(
        Panel(
            main_grid,
            title=title_text,
            border_style=status_style,
            expand=False,
        )
    )
    console.print()


def render_saved_run_detail(run: dict[str, Any]) -> None:
    """Renders a standalone card for an existing saved run dictionary."""
    if run.get("type") == "stress":
        status = run.get("status", "unknown").upper()
        status_style = "green" if status == "SUCCESS" else "red"

        title_text = Text()
        title_text.append("🔥 Saved Concurrency Stress Report: ", style="bold cyan")
        title_text.append(f"{run.get('provider', '').upper()} ", style="bold white")
        title_text.append(f"({run.get('concurrency_level', 1)} Workers)", style="dim")

        summary_table = Table.grid(padding=(0, 2))
        summary_table.add_column("Key", style="dim")
        summary_table.add_column("Value", style="bold")

        summary_table.add_row("Run ID", run.get("id", "N/A"))
        summary_table.add_row("Saved At", run.get("saved_at", "N/A"))
        summary_table.add_row("Status", f"[{status_style}]{status}[/{status_style}]")
        summary_table.add_row("Total Requests", str(run.get("total_requests", 0)))
        summary_table.add_row("Successful", f"[green]{run.get('successful_requests', 0)}[/green]")
        summary_table.add_row("Failed", f"[red]{run.get('failed_requests', 0)}[/red]")
        summary_table.add_row("Timed Out", f"[yellow]{run.get('timed_out_requests', 0)}[/yellow]")
        summary_table.add_row(
            "Rate Limited (429)", f"[magenta]{run.get('rate_limited_requests', 0)}[/magenta]"
        )
        summary_table.add_row("Duration", f"{run.get('wall_clock_duration_s', 0.0):.2f} s")
        summary_table.add_row(
            "Aggregate Decode TPS",
            f"[bold yellow]{run.get('aggregate_decode_tps', 0.0):.2f} tok/s[/bold yellow]",
        )
        summary_table.add_row("Mean Worker Decode TPS", f"{run.get('mean_worker_decode_tps', 0.0):.2f} tok/s")
        if run.get("mean_ttft_ms") is not None:
            summary_table.add_row("Mean TTFT", f"{run.get('mean_ttft_ms', 0.0):.1f} ms")
        if run.get("p95_ttft_ms") is not None:
            summary_table.add_row("p95 TTFT", f"{run.get('p95_ttft_ms', 0.0):.1f} ms")
        if run.get("degradation_percent", 0) > 0:
            summary_table.add_row("TPS Degradation", f"[red]{run.get('degradation_percent'):.1f}%[/red]")

        main_grid = Table.grid(padding=(1, 2))
        main_grid.add_column()
        main_grid.add_row(summary_table)

        sample_errors = run.get("sample_errors", [])
        if sample_errors:
            err_table = Table(title="⚠️ Error Diagnostics", show_header=True, header_style="bold red")
            err_table.add_column("Diagnostics / Failure Details", style="red")
            for err in sample_errors:
                err_table.add_row(err[:140])
            main_grid.add_row(err_table)

        console.print()
        console.print(Panel(main_grid, title=title_text, border_style=status_style, expand=False))
        console.print()
        return

    status = run.get("status", "unknown").upper()
    status_style = "green" if status == "SUCCESS" else "red"

    title_text = Text()
    title_text.append("📜 Saved Benchmark Report: ", style="bold cyan")
    title_text.append(f"{run.get('provider', '').upper()} ", style="bold white")
    title_text.append(f"[{run.get('model', '')}]", style="dim")

    summary_table = Table.grid(padding=(0, 2))
    summary_table.add_column("Key", style="dim")
    summary_table.add_column("Value", style="bold")

    summary_table.add_row("Run ID", run.get("id", "N/A"))
    summary_table.add_row("Saved At", run.get("saved_at", "N/A"))
    summary_table.add_row("Status", f"[{status_style}]{status}[/{status_style}]")
    if run.get("timeout_type") and run.get("timeout_type") != "none":
        summary_table.add_row("Failure Classification", f"[red]{run.get('timeout_type')}[/red]")
    if run.get("error_message"):
        summary_table.add_row("Error", f"[red]{run.get('error_message')}[/red]")

    tps_data = run.get("tps", {})
    timings = run.get("timings", {})
    tokens = run.get("tokens", {})

    metrics_table = Table(title="Performance Telemetry", show_header=True, header_style="bold cyan")
    metrics_table.add_column("Metric", style="dim")
    metrics_table.add_column("Value", justify="right", style="bold yellow")
    metrics_table.add_column("Notes", style="dim")

    metrics_table.add_row(
        "Decode TPS", f"{tps_data.get('decode_tps', 0.0):.2f} tok/s", "Raw generation speed"
    )
    metrics_table.add_row(
        "End-to-End TPS", f"{tps_data.get('e2e_tps', 0.0):.2f} tok/s", "Perceived round-trip velocity"
    )
    if timings.get("ttft_ms") is not None:
        metrics_table.add_row("TTFT", f"{timings.get('ttft_ms', 0.0):.1f} ms", "Time to First Token")
    if timings.get("total_duration_ms") is not None:
        metrics_table.add_row(
            "Total Duration", f"{timings.get('total_duration_ms', 0.0):.1f} ms", "Wall-clock request duration"
        )
    if timings.get("itl_p50_ms"):
        metrics_table.add_row(
            "ITL p50 / p90 / p99",
            f"{timings.get('itl_p50_ms', 0.0):.1f} / {timings.get('itl_p90_ms', 0.0):.1f} / {timings.get('itl_p99_ms', 0.0):.1f} ms",
            "Delay percentiles",
        )
    if timings.get("jitter_ms"):
        metrics_table.add_row(
            "Jitter (RFC 3550)", f"{timings.get('jitter_ms', 0.0):.2f} ms", "Arrival time variance"
        )

    tokens_table = Table(title="Token Accounting", show_header=True, header_style="bold magenta")
    tokens_table.add_column("Type", style="dim")
    tokens_table.add_column("Count", justify="right", style="bold")
    tokens_table.add_row("Input Tokens", str(tokens.get("input_tokens", 0)))
    tokens_table.add_row("Output Tokens", str(tokens.get("output_tokens", 0)))
    tokens_table.add_row("Generated Tokens", str(tokens.get("generated_tokens", 0)))
    tokens_table.add_row("Total Processed", str(tokens.get("total_tokens", 0)))
    tokens_table.add_row("Cache Hit Rate", f"{tokens.get('cache_hit_rate', 0.0) * 100:.1f}%")

    main_grid = Table.grid(padding=(1, 2))
    main_grid.add_column()
    main_grid.add_row(summary_table)
    main_grid.add_row(metrics_table)
    main_grid.add_row(tokens_table)

    console.print()
    console.print(
        Panel(
            main_grid,
            title=title_text,
            border_style=status_style,
            expand=False,
        )
    )
    console.print()


def render_runs_table(runs: list[dict[str, Any]]) -> None:
    """Renders a clean table summarizing saved benchmark runs."""
    if not runs:
        console.print("[yellow]No saved runs found.[/yellow]")
        return

    table = Table(title="Saved Benchmark Reports", show_header=True, header_style="bold cyan")
    table.add_column("#", justify="right", style="dim", width=4)
    table.add_column("Saved At", style="cyan")
    table.add_column("Provider", style="bold white")
    table.add_column("Model / Task", style="blue")
    table.add_column("Status", justify="center")
    table.add_column("Decode TPS", justify="right", style="bold green")
    table.add_column("TTFT", justify="right", style="yellow")
    table.add_column("Run ID", style="dim")

    for idx, r in enumerate(runs, 1):
        status_val = r.get("status", "unknown").upper()
        status_style = "green" if status_val == "SUCCESS" else "red"
        is_stress = r.get("type") == "stress"

        if is_stress:
            prov_str = f"{r.get('provider')} [stress {r.get('concurrency_level', 1)}w]"
            model_str = f"{r.get('total_requests', 0)} reqs"
            status_display = f"[{status_style}]{status_val} ({r.get('successful_requests', 0)}/{r.get('total_requests', 0)})[/{status_style}]"
            tps_val = r.get("aggregate_decode_tps", 0.0)
            ttft_val = r.get("mean_ttft_ms")
        else:
            prov_str = r.get("provider", "N/A")
            model_str = r.get("model", "default")[:24]
            status_display = f"[{status_style}]{status_val}[/{status_style}]"
            tps_val = r.get("tps", {}).get("decode_tps", 0.0)
            ttft_val = r.get("timings", {}).get("ttft_ms")

        ttft_str = f"{ttft_val:.1f}ms" if ttft_val is not None else "N/A"

        table.add_row(
            str(idx),
            r.get("saved_at", "N/A"),
            prov_str,
            model_str,
            status_display,
            f"{tps_val:.1f} tok/s",
            ttft_str,
            r.get("id", "N/A")[:16],
        )

    console.print(table)


def render_concurrency_report(report: ConcurrencyReport) -> None:
    """Renders multi-agent stress benchmark and throughput results."""
    title = f"⚡ Concurrency Stress Report (Workers: {report.concurrency_level})"
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right", style="bold")

    table.add_row("Total Requests", str(report.total_requests))
    table.add_row("Successful Requests", f"[green]{report.successful_requests}[/green]")
    table.add_row("Failed Requests", f"[red]{report.failed_requests}[/red]")
    table.add_row("Timed Out Requests", f"[yellow]{report.timed_out_requests}[/yellow]")
    table.add_row("Rate Limited (429)", f"[magenta]{report.rate_limited_requests}[/magenta]")
    table.add_row("Wall-Clock Duration", f"{report.wall_clock_duration_s:.2f} s")
    table.add_row(
        "Aggregate Decode TPS", f"[bold yellow]{report.aggregate_decode_tps:.2f} tok/s[/bold yellow]"
    )
    table.add_row("Mean Worker Decode TPS", f"{report.mean_worker_decode_tps:.2f} tok/s")
    table.add_row("Mean TTFT", f"{report.mean_ttft_ms:.1f} ms")
    table.add_row("p95 TTFT", f"{report.p95_ttft_ms:.1f} ms")
    table.add_row("p99 TTFT", f"{report.p99_ttft_ms:.1f} ms")

    if report.degradation_percent > 0:
        table.add_row("TPS Degradation", f"[red]{report.degradation_percent:.1f}%[/red]")

    main_grid = Table.grid(padding=(1, 2))
    main_grid.add_column()
    main_grid.add_row(table)

    if report.failed_requests > 0 or report.timed_out_requests > 0 or report.rate_limited_requests > 0:
        err_table = Table(title="⚠️ Failure Diagnostics", show_header=True, header_style="bold red")
        err_table.add_column("Type", style="yellow", width=16)
        err_table.add_column("Count", justify="right", style="bold", width=8)
        err_table.add_column("Sample Reason", style="red")

        error_counts: dict[tuple[str, str], int] = {}
        for r in report.results:
            if r.status != BenchmarkStatus.SUCCESS:
                tt_val = r.timeout_type.value if r.timeout_type else "unknown"
                msg_val = r.error_message or "Unknown failure"
                key = (tt_val, msg_val)
                error_counts[key] = error_counts.get(key, 0) + 1

        for (tt, msg), count in error_counts.items():
            err_table.add_row(tt.upper(), str(count), msg[:120])

        main_grid.add_row(err_table)

    border_color = (
        "green"
        if report.failed_requests == 0 and report.successful_requests > 0
        else "yellow"
        if report.successful_requests > 0
        else "red"
    )
    console.print(Panel(main_grid, border_style=border_color, expand=False))


def render_opencode_sessions(sessions: list[OpenCodeSessionDetail]) -> None:
    """Renders historical OpenCode sessions from SQLite database."""
    if not sessions:
        console.print("[yellow]No recent OpenCode sessions found in local SQLite database.[/yellow]")
        return

    table = Table(title="OpenCode Local Database Session History", show_header=True, header_style="bold cyan")
    table.add_column("Date", style="dim")
    table.add_column("Title / Session", style="bold")
    table.add_column("Model", style="cyan")
    table.add_column("Tokens", justify="right")
    table.add_column("Decode TPS", justify="right", style="bold green")
    table.add_column("E2E TPS", justify="right", style="yellow")
    table.add_column("TTFT", justify="right")
    table.add_column("Cache Hit", justify="right")

    for s in sessions:
        date_str = (
            time.strftime("%Y-%m-%d %H:%M", time.localtime(s.time_created_ms / 1000.0))
            if s.time_created_ms
            else "N/A"
        )
        ttft_str = f"{s.timings.ttft_ms:.1f}ms" if s.timings.ttft_ms is not None else "N/A"
        cache_str = f"{s.tokens.cache_hit_rate * 100:.0f}%"

        table.add_row(
            date_str,
            s.title[:25],
            s.model[:20],
            str(s.tokens.generated_tokens),
            f"{s.tps.decode_tps:.1f} tok/s" if s.tps.decode_tps > 0 else "-",
            f"{s.tps.e2e_tps:.1f} tok/s" if s.tps.e2e_tps > 0 else "-",
            ttft_str,
            cache_str,
        )

    console.print(table)


def export_report_to_json(report: Any, path: Path) -> None:
    """Exports any report or model to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        if hasattr(report, "model_dump"):
            json.dump(report.model_dump(), f, indent=2)
        else:
            json.dump(report, f, indent=2)
    console.print(f"[green]Exported report to {path}[/green]")
