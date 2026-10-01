"""Clean interactive terminal user interface for Agent-TPS.

Inspired by modern developer CLIs (Grok, Claude, OpenCode):
- Instant fuzzy model search across all coding agent drivers
- Standalone single report viewer with dynamic selection
- Subagent concurrency load testing
- Privacy-first metric storage with automatic chat purging
- Clear reports management
"""

from __future__ import annotations

import asyncio

from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text

from agent_tps.core.concurrency import ConcurrencyRunner
from agent_tps.providers.dispatcher import get_runner_for_provider
from agent_tps.providers.opencode import OpenCodeDBReader
from agent_tps.providers.registry import PROVIDERS_CATALOG, get_all_models_flat
from agent_tps.storage.store import BenchmarkStorage
from agent_tps.ui.reporter import (
    render_benchmark_result,
    render_concurrency_report,
    render_opencode_sessions,
    render_saved_run_detail,
)

console = Console()
storage = BenchmarkStorage()


def print_banner() -> None:
    """Prints the sleek Agent-TPS header."""
    header_text = Text()
    header_text.append("⚡ Agent-TPS ", style="bold cyan")
    header_text.append("v0.2.0\n", style="dim")
    header_text.append(
        "Coding Agent Velocity & Concurrency Profiler\n",
        style="bold white",
    )
    header_text.append(
        "Engines: OpenCode · Cursor · Grok · Antigravity · Codex · Claude",
        style="dim cyan",
    )
    console.print(Panel(header_text, border_style="cyan", expand=False))


def select_model_interactive() -> tuple[str, str]:
    """Allows user to either search globally via fuzzy finder or select by provider."""
    mode = inquirer.select(
        message="Select Model Selection Mode:",
        choices=[
            Choice("fuzzy", "🔍 Search All Models (Global Fuzzy Finder)"),
            Choice("provider", "🏢 Filter by Agent Provider First"),
        ],
        default="fuzzy",
    ).execute()

    if mode == "fuzzy":
        all_models = get_all_models_flat()
        choices = [
            Choice(
                value=(m["provider"], m["model_id"]),
                name=m["display"],
            )
            for m in all_models
        ]
        choices.append(Choice(value=("custom", "custom"), name="✍️ Custom Model Slug..."))

        selected = inquirer.fuzzy(
            message="Type to search model across all agent engines:",
            choices=choices,
        ).execute()

        if selected == ("custom", "custom"):
            prov = inquirer.select(
                message="Select Target Provider:",
                choices=list(PROVIDERS_CATALOG.keys()),
            ).execute()
            custom_slug = inquirer.text(message="Enter Custom Model Slug:").execute().strip()
            return prov, custom_slug
        return selected

    else:
        provider_choices = [
            Choice(
                value=p_id,
                name=f"{p_info.name} - {p_info.description} ({'Installed' if p_info.is_installed else 'Not in PATH'})",
            )
            for p_id, p_info in PROVIDERS_CATALOG.items()
        ]
        provider = inquirer.select(
            message="Select Provider / Agent Engine:",
            choices=provider_choices,
        ).execute()

        p_info = PROVIDERS_CATALOG[provider]
        model_choices = [
            Choice(value=m.id, name=f"{m.name} [{m.id}] - {m.description}") for m in p_info.models
        ]
        model_choices.append(Choice(value="custom", name="✍️ Custom Model Slug..."))

        selected_model = inquirer.select(
            message=f"Select {p_info.name} Model:",
            choices=model_choices,
        ).execute()

        if selected_model == "custom":
            selected_model = inquirer.text(message="Enter Model Slug:").execute().strip()

        return provider, selected_model


def interactive_benchmark() -> None:
    """Executes a single benchmark run interactively with live streaming."""
    console.print("\n[bold cyan]─── 🚀 Interactive Benchmark ───[/bold cyan]")

    default_prompt = "Explain concurrency vs parallelism in 2 lines."
    prompt = (
        inquirer.text(
            message="Enter Prompt (hit Enter for default):",
            default=default_prompt,
        )
        .execute()
        .strip()
    )
    if not prompt:
        prompt = default_prompt

    provider, model = select_model_interactive()

    console.print(f"\n[dim]Initializing {provider.upper()} ({model})...[/dim]")
    runner = get_runner_for_provider(provider)

    full_output: list[str] = []
    chunk_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("[yellow]Awaiting first token...[/yellow]", total=None)

        def on_chunk(chunk: str):
            nonlocal chunk_count
            chunk_count += 1
            full_output.append(chunk)
            preview = chunk.replace("\n", " ").strip()[:25]
            progress.update(task, description=f"[green]Streaming ({chunk_count} toks): {preview}...[/green]")

        result = asyncio.run(
            runner.run_prompt(
                prompt=prompt,
                model=model,
                on_chunk=on_chunk,
            )
        )

    # Standalone result rendering
    render_benchmark_result(result)

    # Save metrics without raw prompt or chat text
    saved_path = storage.save_run(result)
    console.print(f"[green]✓ Performance metrics saved to {saved_path.name}[/green]")
    console.print("[dim]✓ Chat session wiped cleanly (zero chat history stored).[/dim]\n")


def interactive_stress_test() -> None:
    """Runs a concurrent subagents stress benchmark."""
    console.print("\n[bold cyan]─── 🔥 Subagent Concurrency Stress Test ───[/bold cyan]")

    provider_choices = [Choice(k, PROVIDERS_CATALOG[k].name) for k in PROVIDERS_CATALOG]
    target = inquirer.select(
        message="Select Benchmark Target:",
        choices=provider_choices,
        default="opencode",
    ).execute()

    concurrency_str = inquirer.select(
        message="Number of Concurrent Workers / Subagents:",
        choices=["2", "4", "8", "16"],
        default="4",
    ).execute()
    concurrency = int(concurrency_str)

    total_str = (
        inquirer.text(
            message="Total Number of Requests:",
            default=str(concurrency * 2),
        )
        .execute()
        .strip()
    )
    total = int(total_str) if total_str.isdigit() else (concurrency * 2)

    prompt = (
        inquirer.text(
            message="Stress Prompt:",
            default="Output a quick 1-sentence Python tip.",
        )
        .execute()
        .strip()
    )

    runner = get_runner_for_provider(target)
    orchestrator = ConcurrencyRunner(runner)

    console.print(f"\n[cyan]Spawning {concurrency} concurrent workers (Total: {total} runs)...[/cyan]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("[yellow]Running concurrency pool...[/yellow]", total=total)

        def on_complete(_, completed: int, total_reqs: int):
            progress.update(
                task,
                completed=completed,
                description=f"[green]Completed {completed}/{total_reqs} workers...[/green]",
            )

        report = asyncio.run(
            orchestrator.run_batch(
                concurrency=concurrency,
                total_requests=total,
                prompt=prompt,
                on_complete=on_complete,
            )
        )

    render_concurrency_report(report)


def interactive_view_runs() -> None:
    """Dynamic standalone report inspector: pick a run to view it standalone."""
    runs = storage.list_runs(limit=30)
    if not runs:
        console.print("[yellow]No saved runs found.[/yellow]\n")
        return

    choices = []
    for r in runs:
        date_str = r.get("saved_at", "N/A")
        prov = r.get("provider", "N/A")
        mod = r.get("model", "N/A")
        tps = r.get("tps", {}).get("decode_tps", 0.0)
        ttft = r.get("timings", {}).get("ttft_ms")
        ttft_str = f"{ttft:.0f}ms" if ttft is not None else "N/A"
        label = f"{date_str} │ {prov:<10} │ {mod:<20} │ {tps:>5.1f} tok/s │ TTFT {ttft_str}"
        choices.append(Choice(value=r.get("id"), name=label))

    choices.append(Choice(value="back", name="⬅️ Back to Main Menu"))

    selected_id = inquirer.select(
        message="Select a Report to View Standalone:",
        choices=choices,
    ).execute()

    if selected_id == "back":
        return

    run_data = storage.get_run(selected_id)
    if run_data:
        render_saved_run_detail(run_data)

        action = inquirer.select(
            message=f"Action for report {selected_id[:16]}:",
            choices=[
                Choice("keep", "⬅️ Keep and return"),
                Choice("delete", "🗑️ Delete this report"),
            ],
            default="keep",
        ).execute()

        if action == "delete":
            storage.delete_run(selected_id)
            console.print(f"[green]Report {selected_id[:16]} deleted.[/green]\n")


def interactive_clear_reports() -> None:
    """Allows deleting a single report or clearing all saved reports."""
    action = inquirer.select(
        message="Clear Reports Menu:",
        choices=[
            Choice("all", "🗑️ Delete ALL Saved Reports"),
            Choice("cancel", "⬅️ Cancel"),
        ],
        default="cancel",
    ).execute()

    if action == "all":
        confirm = inquirer.confirm(
            message="Are you sure you want to delete ALL benchmark reports?",
            default=False,
        ).execute()
        if confirm:
            count = storage.clear_all_runs()
            console.print(f"[green]Deleted {count} benchmark reports.[/green]\n")


def interactive_opencode_history() -> None:
    """Inspects past OpenCode sessions from local SQLite."""
    reader = OpenCodeDBReader()
    if not reader.is_available():
        console.print(
            "[yellow]OpenCode database not found at default location (~/.local/share/opencode/opencode.db).[/yellow]\n"
        )
        return

    sessions = reader.get_recent_sessions(limit=15)
    render_opencode_sessions(sessions)


def run_interactive_tui() -> None:
    """Main loop for the Agent-TPS interactive terminal interface."""
    while True:
        print_banner()
        choice = inquirer.select(
            message="What would you like to do?",
            choices=[
                Choice("bench", "⚡ Run Benchmark (Prompt, Model Fuzzy Finder, Live Speedometer)"),
                Choice("stress", "🔥 Subagent Stress Test (Concurrency & Rate Limits)"),
                Choice("runs", "📜 Saved Reports (Dynamic Standalone Inspector)"),
                Choice("clear", "🗑️ Clear / Delete Reports"),
                Choice("opencode_db", "📂 OpenCode Local History (SQLite Telemetry)"),
                Choice("exit", "🚪 Exit"),
            ],
            default="bench",
        ).execute()

        if choice == "bench":
            interactive_benchmark()
        elif choice == "stress":
            interactive_stress_test()
        elif choice == "runs":
            interactive_view_runs()
        elif choice == "clear":
            interactive_clear_reports()
        elif choice == "opencode_db":
            interactive_opencode_history()
        elif choice == "exit":
            console.print("[dim]Goodbye![/dim]")
            break
