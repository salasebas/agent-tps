"""Command-line interface for Agent-TPS."""

from __future__ import annotations

import asyncio
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
import typer

from agent_tps.core.concurrency import ConcurrencyRunner
from agent_tps.providers.dispatcher import get_runner_for_provider
from agent_tps.providers.opencode import OpenCodeDBReader
from agent_tps.storage.store import BenchmarkStorage
from agent_tps.ui.reporter import (
    export_report_to_json,
    render_benchmark_result,
    render_concurrency_report,
    render_opencode_sessions,
    render_runs_table,
    render_saved_run_detail,
)

app = typer.Typer(
    name="agent-tps",
    help="High-Velocity Benchmark & Concurrency Profiler for Coding Agents (OpenCode, Cursor, Grok, Antigravity, Codex, Claude).",
    add_completion=False,
    invoke_without_command=True,
)

console = Console()
storage = BenchmarkStorage()


@app.callback()
def main_callback(ctx: typer.Context):
    """If no subcommand is passed, launch the interactive TUI."""
    if ctx.invoked_subcommand is None:
        from agent_tps.ui.interactive import run_interactive_tui

        run_interactive_tui()


@app.command("ui")
def ui_command():
    """Launch the interactive terminal UI with fuzzy model search and reports inspector."""
    from agent_tps.ui.interactive import run_interactive_tui

    run_interactive_tui()


@app.command("bench")
def bench_command(
    prompt: str = typer.Option(
        "Explain concurrency vs parallelism in 2 lines.",
        "--prompt",
        "-p",
        help="Prompt to execute",
    ),
    provider: str = typer.Option(
        "opencode",
        "--provider",
        help="Target agent engine: opencode, cursor, grok, antigravity, codex, claude",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Model slug or name",
    ),
    ttft_timeout: float = typer.Option(15.0, "--ttft-timeout", help="Max seconds to wait for first token"),
    stall_timeout: float = typer.Option(
        10.0, "--stall-timeout", help="Max seconds between consecutive tokens"
    ),
    deadline_timeout: float = typer.Option(
        60.0, "--deadline-timeout", help="Total request timeout in seconds"
    ),
    output_json: Path | None = typer.Option(None, "--output-json", "-o", help="Export result to JSON"),
):
    """Runs a benchmark prompt against an agent engine, measuring TPS, TTFT, and latency."""
    console.print(f"[bold cyan]Launching TokPulse benchmark on {provider.upper()}...[/bold cyan]")
    runner = get_runner_for_provider(provider)

    full_output: list[str] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("[yellow]Streaming tokens...[/yellow]", total=None)

        def on_chunk(chunk: str):
            full_output.append(chunk)
            preview = chunk.replace("\n", " ").strip()[:20]
            progress.update(task, description=f"[green]Streaming: {preview}...[/green]")

        result = asyncio.run(
            runner.run_prompt(
                prompt=prompt,
                model=model,
                ttft_timeout_s=ttft_timeout,
                stall_timeout_s=stall_timeout,
                deadline_timeout_s=deadline_timeout,
                on_chunk=on_chunk,
            )
        )

    render_benchmark_result(result)
    saved_path = storage.save_run(result)
    console.print(f"[green]Saved metrics to {saved_path.name}[/green]")
    console.print("[dim]Chat session wiped cleanly (zero chat history stored).[/dim]")

    if output_json:
        export_report_to_json(result, output_json)


@app.command("stress")
def stress_command(
    target: str = typer.Option(
        "opencode",
        "--target",
        "-t",
        help="Target agent: opencode, cursor, grok, antigravity, codex, claude",
    ),
    concurrency: int = typer.Option(4, "--concurrency", "-c", help="Number of concurrent subagents/workers"),
    total: int = typer.Option(8, "--total", "-n", help="Total number of requests to execute"),
    sweep: bool = typer.Option(False, "--sweep", help="Perform progressive concurrency sweep (1, 2, 4, 8)"),
    prompt: str = typer.Option("Output a quick 1-sentence Python tip.", "--prompt", "-p"),
    model: str | None = typer.Option(None, "--model", "-m"),
    output_json: Path | None = typer.Option(None, "--output-json", "-o"),
):
    """Stress tests concurrency scaling, throughput saturation, and rate limits."""
    runner = get_runner_for_provider(target)
    orchestrator = ConcurrencyRunner(runner)

    if sweep:
        concurrency_levels = [1, 2, 4, 8]
        console.print(f"[bold cyan]Running concurrency sweep across {concurrency_levels}...[/bold cyan]")
        reports = asyncio.run(
            orchestrator.run_sweep(
                concurrency_levels=concurrency_levels,
                requests_per_level=total // len(concurrency_levels) or 2,
                prompt=prompt,
                model=model,
            )
        )
        for r in reports:
            render_concurrency_report(r)
            storage.save_concurrency_run(r, target=target, model=model)
        if output_json:
            export_report_to_json([r.model_dump() for r in reports], output_json)
    else:
        console.print(
            f"[bold cyan]Launching concurrency stress test (Workers: {concurrency}, Total: {total})...[/bold cyan]"
        )
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("[yellow]Running requests...[/yellow]", total=total)

            def on_comp(_, completed: int, total_reqs: int):
                progress.update(
                    task,
                    completed=completed,
                    description=f"[green]Completed {completed}/{total_reqs}...[/green]",
                )

            report = asyncio.run(
                orchestrator.run_batch(
                    concurrency=concurrency,
                    total_requests=total,
                    prompt=prompt,
                    model=model,
                    on_complete=on_comp,
                )
            )

        render_concurrency_report(report)
        saved_path = storage.save_concurrency_run(report, target=target, model=model)
        console.print(f"[green]✓ Stress test report saved to {saved_path.name}[/green]")
        if output_json:
            export_report_to_json(report, output_json)


@app.command("runs")
def list_runs(
    limit: int = typer.Option(20, "--limit", "-l", help="Number of runs to list"),
):
    """Lists saved benchmark runs stored locally."""
    runs = storage.list_runs(limit=limit)
    render_runs_table(runs)


@app.command("view")
def view_run(
    run_id: str | None = typer.Argument(
        None,
        help="Run ID or prefix to view standalone. If omitted, launches interactive picker.",
    ),
):
    """View a single benchmark report standalone on the screen."""
    if not run_id:
        from agent_tps.ui.interactive import interactive_view_runs

        interactive_view_runs()
        return

    run_data = storage.get_run(run_id)
    if not run_data:
        console.print(f"[red]Error: Run with ID or prefix '{run_id}' not found.[/red]")
        raise typer.Exit(code=1)

    render_saved_run_detail(run_data)


@app.command("delete")
def delete_run(
    run_id: str = typer.Argument(..., help="Run ID or filename prefix to delete"),
):
    """Delete a specific saved benchmark report."""
    success = storage.delete_run(run_id)
    if success:
        console.print(f"[green]Report '{run_id}' deleted successfully.[/green]")
    else:
        console.print(f"[red]Report '{run_id}' not found.[/red]")


@app.command("clear")
def clear_runs(
    yes: bool = typer.Option(False, "--yes", "-y", help="Confirm deletion without prompting"),
):
    """Wipes all saved benchmark reports."""
    if not yes:
        confirm = typer.confirm("Are you sure you want to delete all saved reports?")
        if not confirm:
            console.print("[yellow]Aborted.[/yellow]")
            return

    count = storage.clear_all_runs()
    console.print(f"[green]Deleted {count} benchmark reports.[/green]")


@app.command("opencode-history")
def opencode_history(
    limit: int = typer.Option(15, "--limit", "-l", help="Number of recent sessions to inspect"),
):
    """Inspects OpenCode session history from local opencode.db with computed TPS and TTFT."""
    reader = OpenCodeDBReader()
    if not reader.is_available():
        console.print("[red]OpenCode SQLite database not found at ~/.local/share/opencode/opencode.db[/red]")
        raise typer.Exit(code=1)

    sessions = reader.get_recent_sessions(limit=limit)
    render_opencode_sessions(sessions)


def main():
    app()


if __name__ == "__main__":
    main()
