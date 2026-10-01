from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from agent_tps_bench.concurrency import ConcurrencyRunner
from agent_tps_bench.llm_stream_runner import LLMStreamRunner
from agent_tps_bench.models import BenchmarkResult
from agent_tps_bench.opencode_reader import OpenCodeDBReader
from agent_tps_bench.opencode_runner import OpenCodeRunner
from agent_tps_bench.opencode_server_runner import OpenCodeServerRunner
from agent_tps_bench.reporter import (
    console,
    export_report_to_json,
    render_benchmark_result,
    render_concurrency_report,
    render_concurrency_sweep,
    render_opencode_sessions,
)

app = typer.Typer(
    name="agent-tps",
    help="High-precision benchmark suite for Agent and LLM TPS, TTFT, Timeouts, Concurrency, and Rate Limits.",
    add_completion=False,
)


@app.command("opencode-bench")
def opencode_bench(
    prompt: str = typer.Option(
        "Escribe una funcion en Python para invertir una lista enlazada y explica su complejidad temporal.",
        "--prompt",
        "-p",
        help="Prompt to execute with OpenCode",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model slug to use (e.g. opencode/longcat-2.5-preview-free)",
    ),
    ttft_timeout: float = typer.Option(15.0, "--ttft-timeout", help="Max seconds to wait for first token"),
    stall_timeout: float = typer.Option(10.0, "--stall-timeout", help="Max seconds between consecutive tokens"),
    deadline_timeout: float = typer.Option(60.0, "--deadline-timeout", help="Total request execution timeout in seconds"),
    output_json: Optional[Path] = typer.Option(None, "--output-json", "-o", help="Export result to JSON"),
):
    """Runs an active prompt against OpenCode, measuring real-time TPS, TTFT, and timeout classification."""
    console.print(f"[bold cyan]Launching OpenCode TPS benchmark...[/bold cyan]")
    runner = OpenCodeRunner()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("[yellow]Streaming tokens from OpenCode...[/yellow]", total=None)

        def on_chunk(chunk: str):
            progress.update(task, description=f"[green]Streaming: {chunk[:20].strip()}...[/green]")

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
    if output_json:
        export_report_to_json(result, output_json)
        console.print(f"[green]Saved JSON report to {output_json}[/green]")


@app.command("opencode-history")
def opencode_history(
    limit: int = typer.Option(15, "--limit", "-l", help="Number of recent sessions to inspect"),
    session_id: Optional[str] = typer.Option(None, "--session", "-s", help="Inspect a specific session ID"),
    db_path: Optional[Path] = typer.Option(None, "--db", help="Path to opencode.db"),
    output_json: Optional[Path] = typer.Option(None, "--output-json", "-o", help="Export results to JSON"),
):
    """Inspects OpenCode session history from opencode.db, displaying computed TPS, TTFT, and cache ratios."""
    reader = OpenCodeDBReader(db_path=db_path)
    if not reader.is_available():
        console.print(f"[red]OpenCode database not found at {reader.db_path}[/red]")
        raise typer.Exit(code=1)

    if session_id:
        detail = reader.get_session_by_id(session_id)
        if not detail:
            console.print(f"[red]Session {session_id} not found[/red]")
            raise typer.Exit(code=1)
        res = reader.to_benchmark_result(detail)
        render_benchmark_result(res)
        if output_json:
            export_report_to_json(res, output_json)
    else:
        sessions = reader.get_recent_sessions(limit=limit)
        render_opencode_sessions(sessions)
        if output_json:
            export_report_to_json(sessions, output_json)


@app.command("stream-bench")
def stream_bench(
    model: str = typer.Option("gpt-4o", "--model", "-m", help="Model name"),
    prompt: str = typer.Option(
        "Cuenta una historia corta de 3 parrafos sobre un robot explorador.",
        "--prompt",
        "-p",
        help="Prompt to send",
    ),
    provider: str = typer.Option("openai", "--provider", help="Preset provider (openai, openrouter, groq, cerebras, ollama)"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Custom OpenAI-compatible base URL"),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", envvar="OPENAI_API_KEY", help="API Key"),
    connect_timeout: float = typer.Option(8.0, "--connect-timeout", help="Connection timeout in seconds"),
    ttft_timeout: float = typer.Option(12.0, "--ttft-timeout", help="TTFT timeout in seconds"),
    stall_timeout: float = typer.Option(5.0, "--stall-timeout", help="Stall/heartbeat timeout in seconds"),
    deadline_timeout: float = typer.Option(60.0, "--deadline-timeout", help="Deadline timeout in seconds"),
    output_json: Optional[Path] = typer.Option(None, "--output-json", "-o", help="Export result to JSON"),
):
    """Direct streaming benchmark against any OpenAI-compatible API endpoint."""
    console.print(f"[bold cyan]Connecting to {provider} stream...[/bold cyan]")
    runner = LLMStreamRunner(base_url=base_url, api_key=api_key, provider=provider)

    result = asyncio.run(
        runner.run_stream(
            model=model,
            prompt=prompt,
            connect_timeout_s=connect_timeout,
            ttft_timeout_s=ttft_timeout,
            stall_timeout_s=stall_timeout,
            deadline_timeout_s=deadline_timeout,
        )
    )

    render_benchmark_result(result)
    if output_json:
        export_report_to_json(result, output_json)
        console.print(f"[green]Saved JSON report to {output_json}[/green]")


@app.command("stress")
def stress_benchmark(
    target: str = typer.Option("opencode", "--target", "-t", help="Benchmark target: 'opencode' or 'stream'"),
    concurrency: int = typer.Option(4, "--concurrency", "-c", help="Number of concurrent subagents/workers"),
    total: int = typer.Option(8, "--total", "-n", help="Total number of requests to execute"),
    sweep: bool = typer.Option(False, "--sweep", help="Perform progressive concurrency sweep (1, 2, 4, 8)"),
    prompt: str = typer.Option("Responde en 2 lineas que es un puntero en C.", "--prompt", "-p"),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    base_url: Optional[str] = typer.Option(None, "--base-url"),
    api_key: Optional[str] = typer.Option(None, "--api-key", envvar="OPENAI_API_KEY"),
    output_json: Optional[Path] = typer.Option(None, "--output-json", "-o"),
):
    """Stress tests concurrency scaling, throughput saturation, and rate limits."""
    if target.lower() in ("opencode-server", "opencode"):
        # For multi-agent concurrency, OpenCodeServerRunner handles SQLite serialization smoothly
        runner = OpenCodeServerRunner()
    else:
        runner = LLMStreamRunner(base_url=base_url, api_key=api_key, provider="custom")
    concurrency_orchestrator = ConcurrencyRunner(runner)

    if sweep:
        levels = [1, 2, 4, 8]
        console.print(f"[bold cyan]Running concurrency sweep across levels {levels}...[/bold cyan]")
        reports = asyncio.run(
            concurrency_orchestrator.run_sweep(
                concurrency_levels=levels,
                requests_per_level=total,
                prompt=prompt,
                model=model,
            )
        )
        render_concurrency_sweep(reports)
        if output_json:
            export_report_to_json(reports, output_json)
    else:
        console.print(f"[bold cyan]Running concurrency batch: {concurrency} workers, {total} requests...[/bold cyan]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[yellow]Executing concurrent requests...[/yellow]", total=total)

            def on_comp(res: BenchmarkResult, completed: int, tot: int):
                progress.update(task, completed=completed)

            report = asyncio.run(
                concurrency_orchestrator.run_batch(
                    concurrency=concurrency,
                    total_requests=total,
                    prompt=prompt,
                    model=model,
                    on_complete=on_comp,
                )
            )

        render_concurrency_report(report)
        if output_json:
            export_report_to_json(report, output_json)


def main():
    app()


if __name__ == "__main__":
    main()
