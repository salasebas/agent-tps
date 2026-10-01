"""Clean interactive terminal user interface for Agent-TPS.

Inspired by modern developer CLIs (Grok, Claude, OpenCode):
- Instant fuzzy model search across all coding agent drivers
- Standalone single report viewer with dynamic selection
- Subagent concurrency load testing with curated heavy benchmark presets
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
from agent_tps.providers.registry import PROVIDERS_CATALOG, get_all_models_flat
from agent_tps.storage.store import BenchmarkStorage
from agent_tps.ui.reporter import (
    render_benchmark_result,
    render_concurrency_report,
    render_saved_run_detail,
)

console = Console()
storage = BenchmarkStorage()

PRESET_PROMPTS: list[dict[str, str]] = [
    {
        "id": "lru",
        "title": "Thread-safe LRU Cache with TTL (Heavy Logic)",
        "prompt": (
            "Write a production-ready, thread-safe LRU Cache in Python. Requirements:\n"
            "1. TTL expiration per key with a background thread cleanup.\n"
            "2. Thread-safety with RLock and atomic hit/miss metrics.\n"
            "3. Comprehensive unit tests covering expiration, eviction order, and race conditions.\n"
            "4. Full typing annotations and docstrings."
        ),
    },
    {
        "id": "rest_api",
        "title": "Async REST API with JWT Auth & Rate Limiter (Heavy Architecture)",
        "prompt": (
            "Implement a production-grade asynchronous REST API microservice in Python using FastAPI.\n"
            "1. JWT authentication middleware with token expiration and refresh.\n"
            "2. Sliding-window rate limiter per client IP / API key.\n"
            "3. Pydantic v2 validation models for User and Auth schemas.\n"
            "4. CRUD routes for users with mock in-memory database and complete pytest fixtures."
        ),
    },
    {
        "id": "parser",
        "title": "Markdown to AST & HTML Compiler from scratch (Deep Generation)",
        "prompt": (
            "Build a complete Markdown-to-HTML parser and AST compiler in Python without regex or external packages.\n"
            "1. Lexer and recursive descent parser that produces an Abstract Syntax Tree (AST).\n"
            "2. Support headers (# to ######), blockquotes, lists (ordered/unordered), fenced code blocks, and bold/italic.\n"
            "3. Tree-walking HTML renderer with proper character escaping.\n"
            "4. Test suite verifying AST node structure and rendered HTML output."
        ),
    },
    {
        "id": "grep",
        "title": "Multi-threaded CLI Grep & File Search Engine (CLI & Threads)",
        "prompt": (
            "Develop a high-performance multi-threaded CLI grep tool in Python using argparse/click.\n"
            "1. Worker thread pool scanning directories recursively with glob and .gitignore filtering.\n"
            "2. Regex pattern search with case-sensitivity and invert match flags.\n"
            "3. Binary file detection and colored match highlights with line numbers and column offsets.\n"
            "4. Concurrency benchmark comparing thread pool sizes (1, 2, 4, 8 workers)."
        ),
    },
    {
        "id": "sql",
        "title": "In-Memory SQL Query Evaluator & Aggregations (Parsing & Aggregations)",
        "prompt": (
            "Create an in-memory SQL query evaluator in Python from scratch.\n"
            "1. Parse standard SQL queries: SELECT, FROM, WHERE, GROUP BY, ORDER BY, LIMIT.\n"
            "2. Support boolean expressions (AND, OR, NOT, =, !=, >, <) in WHERE clauses.\n"
            "3. Implement aggregate functions: COUNT(*), SUM(col), AVG(col), MIN(col), MAX(col).\n"
            "4. Include test cases running complex queries against mock tabular datasets."
        ),
    },
]


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


def select_prompt_interactive(default_custom: str = "Explain concurrency vs parallelism in 2 lines.") -> str:
    """Lets user select between 5 heavy realistic presets or write a custom prompt."""
    choices = [Choice(p["id"], f"{idx}. {p['title']}") for idx, p in enumerate(PRESET_PROMPTS, 1)]
    choices.append(Choice("custom", "✍️  Custom Prompt..."))

    choice = inquirer.select(
        message="Prompt:",
        choices=choices,
        default="lru",
    ).execute()

    if choice == "custom":
        user_prompt = (
            inquirer.text(
                message="Enter Custom Prompt:",
                default=default_custom,
            )
            .execute()
            .strip()
        )
        return user_prompt if user_prompt else default_custom

    for p in PRESET_PROMPTS:
        if p["id"] == choice:
            return p["prompt"]

    return default_custom


def select_model_interactive() -> tuple[str, str]:
    """Allows user to either search globally via fuzzy finder or select by provider."""
    mode = inquirer.select(
        message="Model Search Mode:",
        choices=[
            Choice("fuzzy", "🔍  Search All Models (Fuzzy Finder)"),
            Choice("provider", "🏢  Filter by Provider First"),
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
            message="Search model across engines:",
            choices=choices,
        ).execute()

        if selected == ("custom", "custom"):
            prov = inquirer.select(
                message="Target Provider:",
                choices=list(PROVIDERS_CATALOG.keys()),
            ).execute()
            custom_slug = inquirer.text(message="Enter Model Slug:").execute().strip()
            return prov, custom_slug
        return selected

    else:
        provider_choices = [
            Choice(
                value=p_id,
                name=f"{p_info.name} ({'Installed' if p_info.is_installed else 'Not in PATH'})",
            )
            for p_id, p_info in PROVIDERS_CATALOG.items()
        ]
        provider = inquirer.select(
            message="Provider:",
            choices=provider_choices,
        ).execute()

        p_info = PROVIDERS_CATALOG[provider]
        model_choices = [Choice(value=m.id, name=f"{m.name} [{m.id}]") for m in p_info.models]
        model_choices.append(Choice(value="custom", name="✍️ Custom Model Slug..."))

        selected_model = inquirer.select(
            message=f"{p_info.name} Model:",
            choices=model_choices,
        ).execute()

        if selected_model == "custom":
            selected_model = inquirer.text(message="Enter Model Slug:").execute().strip()

        return provider, selected_model


def interactive_benchmark() -> None:
    """Executes a single benchmark run interactively with live streaming."""
    console.print("\n[bold cyan]─── 🚀 Benchmark ───[/bold cyan]")

    prompt = select_prompt_interactive(default_custom="Explain concurrency vs parallelism in 2 lines.")
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
    console.print("\n[bold cyan]─── 🔥 Stress Test ───[/bold cyan]")

    provider_choices = [
        Choice(
            k,
            f"{PROVIDERS_CATALOG[k].name} ({'Installed' if PROVIDERS_CATALOG[k].is_installed else 'Not in PATH'})",
        )
        for k in PROVIDERS_CATALOG
    ]
    target = inquirer.select(
        message="Target:",
        choices=provider_choices,
        default="opencode",
    ).execute()

    concurrency_str = inquirer.select(
        message="Workers (Subagents):",
        choices=["2", "4", "8", "16"],
        default="4",
    ).execute()
    concurrency = int(concurrency_str)

    total_str = (
        inquirer.text(
            message="Total Requests:",
            default=str(concurrency * 2),
        )
        .execute()
        .strip()
    )
    total = int(total_str) if total_str.isdigit() else (concurrency * 2)

    prompt = select_prompt_interactive(default_custom="Output a quick 1-sentence Python tip.")

    runner = get_runner_for_provider(target)
    orchestrator = ConcurrencyRunner(runner)

    console.print(f"\n[cyan]Spawning {concurrency} workers (Total: {total} runs)...[/cyan]")

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

    # Save stress test report to disk
    saved_path = storage.save_concurrency_run(report, target=target)
    console.print(f"[green]✓ Stress test report saved to {saved_path.name}[/green]\n")


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
        is_stress = r.get("type") == "stress"
        status_val = r.get("status", "N/A")

        if is_stress:
            c_w = r.get("concurrency_level", 1)
            n_reqs = r.get("total_requests", 0)
            tps = r.get("aggregate_decode_tps", 0.0)
            status_tag = f"[{status_val}]"
            label = f"{date_str}   {prov:<11}   {status_tag:<9}   {c_w}w/{n_reqs} reqs   {tps:>5.1f} tok/s"
        else:
            mod = r.get("model", "N/A")
            tps = r.get("tps", {}).get("decode_tps", 0.0)
            ttft = r.get("timings", {}).get("ttft_ms")
            ttft_str = f"{ttft:.0f}ms" if ttft is not None else "-"
            status_tag = f"[{status_val}]"
            label = (
                f"{date_str}   {prov:<11}   {status_tag:<9}   {mod:<18}   {tps:>5.1f} tok/s   TTFT {ttft_str}"
            )

        choices.append(Choice(value=r.get("id"), name=label))

    choices.append(Choice(value="back", name="🔙  Back to Main Menu"))

    selected_id = inquirer.select(
        message="Select Report:",
        choices=choices,
    ).execute()

    if selected_id == "back":
        return

    run_data = storage.get_run(selected_id)
    if run_data:
        render_saved_run_detail(run_data)

        action = inquirer.select(
            message=f"Report {selected_id[:16]}:",
            choices=[
                Choice("keep", "🔙  Back to List"),
                Choice("delete", "🗑️   Delete This Report"),
            ],
            default="keep",
        ).execute()

        if action == "delete":
            storage.delete_run(selected_id)
            console.print(f"[green]Report {selected_id[:16]} deleted.[/green]\n")


def interactive_clear_reports() -> None:
    """Allows deleting a single report or clearing all saved reports."""
    action = inquirer.select(
        message="Clear Reports:",
        choices=[
            Choice("all", "🗑️   Delete ALL Saved Reports"),
            Choice("cancel", "🔙  Cancel"),
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


def run_interactive_tui() -> None:
    """Main loop for the Agent-TPS interactive terminal interface."""
    while True:
        print_banner()
        choice = inquirer.select(
            message="Action:",
            choices=[
                Choice("bench", "⚡  Benchmark"),
                Choice("stress", "🔥  Stress Test"),
                Choice("runs", "📜  Saved Reports"),
                Choice("clear", "🗑️   Clear Reports"),
                Choice("exit", "🚪  Exit"),
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
        elif choice == "exit":
            console.print("[dim]Goodbye![/dim]\n")
            break
