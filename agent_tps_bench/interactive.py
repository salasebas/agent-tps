from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import time
from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.syntax import Syntax
from rich.table import Table

from agent_tps_bench.agent_dispatcher import get_runner_for_provider
from agent_tps_bench.concurrency import ConcurrencyRunner
from agent_tps_bench.models import BenchmarkResult
from agent_tps_bench.opencode_reader import OpenCodeDBReader
from agent_tps_bench.providers_registry import (
    get_all_models_flat,
    get_installed_providers,
    get_models_for_provider,
)
from agent_tps_bench.reporter import (
    console,
    render_benchmark_result,
    render_concurrency_report,
    render_concurrency_sweep,
    render_opencode_sessions,
)
from agent_tps_bench.storage import BenchmarkStorage


storage = BenchmarkStorage()


def render_banner() -> None:
    banner_text = (
        "[bold cyan]⚡ AGENT & LLM TPS BENCHMARK SUITE[/bold cyan] [dim](2026 Edition)[/dim]\n"
        "[italic white]High-precision throughput, TTFT latency, timeout watchdogs & subagent stress testing[/italic white]"
    )
    console.print(Panel(banner_text, border_style="cyan", padding=(1, 2)))


def select_model_globally() -> tuple[str, str]:
    """Allows fuzzy searching across all models and providers simultaneously."""
    all_models = get_all_models_flat()
    choices = [
        Choice(
            value=(m.provider, m.model),
            name=f"{m.display:<45} {m.description}",
        )
        for m in all_models
    ]
    selected = inquirer.fuzzy(
        message="Buscador global de modelos (escribe para filtrar):",
        choices=choices,
        multiselect=False,
    ).execute()
    return selected


def select_provider_then_model() -> tuple[str, str]:
    """Allows selecting a provider first, then choosing a model with fuzzy search."""
    providers = get_installed_providers()
    provider_choices = []
    for p in providers:
        status_tag = "[INSTALADO]" if p.is_installed and p.kind == "agent" else "[API DISPONIBLE]" if p.kind == "api" else "[CLI NO DETECTADO]"
        color = "green" if "[INSTALADO]" in status_tag else "cyan" if "[API" in status_tag else "dim"
        provider_choices.append(
            Choice(
                value=p.name,
                name=f"{p.display:<35} {status_tag}",
            )
        )

    chosen_provider = inquirer.select(
        message="Selecciona un Proveedor o Agente:",
        choices=provider_choices,
    ).execute()

    models = get_models_for_provider(chosen_provider)
    if models:
        model_choices = [
            Choice(
                value=m.model,
                name=f"{m.display:<35} {m.description}",
            )
            for m in models
        ]
        chosen_model = inquirer.fuzzy(
            message=f"Selecciona un modelo para {chosen_provider.upper()} (escribe para buscar):",
            choices=model_choices,
        ).execute()
    else:
        chosen_model = inquirer.text(
            message=f"Escribe el nombre del modelo para {chosen_provider}:",
            default="default",
        ).execute()

    return chosen_provider, chosen_model


def interactive_single_benchmark() -> None:
    """Runs a single prompt benchmark with interactive setup."""
    console.print("\n[bold cyan]─── 🚀 Configurar Benchmark Individual ───[/bold cyan]")

    prompt_choices = [
        "Escribe una función en Python para invertir una lista enlazada y explica su complejidad.",
        "Explica en 2 párrafos la diferencia entre concurrencia y paralelismo.",
        "Genera un struct en Rust con métodos para un buffer circular concurrente.",
        "Responde en 5 palabras qué es la memoria RAM.",
        "✍️  Escribir prompt personalizado...",
    ]

    selected_prompt_choice = inquirer.select(
        message="Selecciona un prompt de prueba o escribe uno nuevo:",
        choices=prompt_choices,
    ).execute()

    if selected_prompt_choice.startswith("✍️"):
        prompt = inquirer.text(message="Ingresa tu prompt:").execute()
    else:
        prompt = selected_prompt_choice

    method_choice = inquirer.select(
        message="¿Cómo deseas seleccionar el modelo?",
        choices=[
            Choice("fuzzy_global", "🔍 Buscador Global de Modelos (escribe 'sonnet', 'flash', 'llama', etc.)"),
            Choice("provider_first", "📁 Elegir Proveedor primero (OpenCode, Codex, Claude, Grok, Groq...)"),
        ],
    ).execute()

    if method_choice == "fuzzy_global":
        provider, model = select_model_globally()
    else:
        provider, model = select_provider_then_model()

    console.print(f"[bold green]✔ Seleccionado:[/bold green] Proveedor=[cyan]{provider}[/cyan] | Modelo=[cyan]{model}[/cyan]")

    custom_timeouts = inquirer.confirm(
        message="¿Deseas personalizar los timeouts (TTFT, Stall, Deadline)?",
        default=False,
    ).execute()

    ttft_timeout = 15.0
    stall_timeout = 10.0
    deadline_timeout = 60.0

    if custom_timeouts:
        ttft_str = inquirer.text(message="TTFT timeout en segundos (primer token):", default="15.0").execute()
        stall_str = inquirer.text(message="Stall timeout en segundos (gap entre tokens):", default="10.0").execute()
        dead_str = inquirer.text(message="Deadline timeout total en segundos:", default="60.0").execute()
        try:
            ttft_timeout = float(ttft_str)
            stall_timeout = float(stall_str)
            deadline_timeout = float(dead_str)
        except ValueError:
            console.print("[yellow]Valores inválidos, usando valores recomendados por defecto.[/yellow]")

    runner = get_runner_for_provider(provider)

    full_output_chunks: list[str] = []
    console.print(f"\n[bold yellow]Iniciando ejecución contra {provider}...[/bold yellow]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("[yellow]Generando tokens...[/yellow]", total=None)

        def on_chunk(chunk: str):
            full_output_chunks.append(chunk)
            progress.update(task, description=f"[green]Streaming: {chunk[:25].strip()}...[/green]")

        result = asyncio.run(
            runner.run_prompt(
                prompt=prompt,
                model=model,
                ttft_timeout_s=ttft_timeout,
                stall_timeout_s=stall_timeout,
                deadline_timeout_s=deadline_timeout,
                on_chunk=on_chunk,
            )
            if hasattr(runner, "run_prompt")
            else runner.run_stream(
                model=model,
                prompt=prompt,
                connect_timeout_s=8.0,
                ttft_timeout_s=ttft_timeout,
                stall_timeout_s=stall_timeout,
                deadline_timeout_s=deadline_timeout,
                on_chunk=on_chunk,
            )
        )

    render_benchmark_result(result)

    # Automatically persist run
    saved_path = storage.save_run(
        result=result,
        prompt=prompt,
        full_output="".join(full_output_chunks),
    )
    console.print(f"[bold green]✔ Resultado guardado automáticamente en:[/bold green] [dim]{saved_path}[/dim]")

    # Option to view full output
    if full_output_chunks or result.raw_response_preview:
        view_text = inquirer.confirm(message="¿Deseas ver el texto completo generado por el modelo?", default=False).execute()
        if view_text:
            text_to_show = "".join(full_output_chunks) or result.raw_response_preview
            console.print(Panel(text_to_show, title="[bold cyan]Respuesta Completa Generada[/bold cyan]", border_style="blue"))


def interactive_stress_benchmark() -> None:
    """Runs a subagent concurrency / stress benchmark interactively."""
    console.print("\n[bold cyan]─── ⚡ Stress Test & Concurrencia de Subagentes ───[/bold cyan]")

    target_choice = inquirer.select(
        message="Selecciona el tipo de objetivo para la prueba de estrés:",
        choices=[
            Choice("opencode", "OpenCode (CLI con retry de SQLite)"),
            Choice("opencode-server", "OpenCode Server (T3 Code style, HTTP concurrente sin bloqueos)"),
            Choice("claude", "Claude Code CLI"),
            Choice("codex", "Codex CLI"),
            Choice("groq", "Groq Cloud API (LPU ultra-rápido)"),
            Choice("openai", "OpenAI API"),
            Choice("openrouter", "OpenRouter"),
        ],
    ).execute()

    models = get_models_for_provider(target_choice)
    if models:
        model = inquirer.fuzzy(
            message="Selecciona el modelo:",
            choices=[Choice(m.model, f"{m.display:<30} {m.description}") for m in models],
        ).execute()
    else:
        model = "default"

    mode_choice = inquirer.select(
        message="¿Qué tipo de prueba de estrés deseas correr?",
        choices=[
            Choice("batch", "Lote Fijo (ej. N trabajadores concurrentes simultáneos)"),
            Choice("sweep", "Barrido Progresivo de Concurrencia (1, 2, 4, 8 trabajadores para curva de degradación)"),
        ],
    ).execute()

    prompt = inquirer.text(
        message="Prompt para los subagentes:",
        default="Responde en 2 líneas qué es un hilo (thread) en sistemas operativos.",
    ).execute()

    runner = get_runner_for_provider(target_choice)
    orchestrator = ConcurrencyRunner(runner)

    if mode_choice == "sweep":
        levels = [1, 2, 4, 8]
        reqs_str = inquirer.text(message="Peticiones por cada nivel de concurrencia:", default="2").execute()
        reqs = int(reqs_str)
        console.print(f"[bold cyan]Iniciando barrido con niveles {levels}...[/bold cyan]")
        reports = asyncio.run(
            orchestrator.run_sweep(
                concurrency_levels=levels,
                requests_per_level=reqs,
                prompt=prompt,
                model=model,
            )
        )
        render_concurrency_sweep(reports)
    else:
        conc_str = inquirer.text(message="Número de subagentes/trabajadores concurrentes:", default="4").execute()
        total_str = inquirer.text(message="Total de solicitudes a ejecutar:", default="8").execute()
        concurrency = int(conc_str)
        total = int(total_str)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[yellow]Ejecutando subagentes concurrentes...[/yellow]", total=total)

            def on_comp(res: BenchmarkResult, completed: int, tot: int):
                progress.update(task, completed=completed)

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


def interactive_view_runs(interactive: bool = True) -> None:
    """Lists saved runs from ~/.agent-tps-bench/runs/ and allows inspecting any run."""
    console.print("\n[bold cyan]─── 📜 Historial de Ejecuciones Guardadas ───[/bold cyan]")
    runs = storage.list_runs(limit=30)
    if not runs:
        console.print("[yellow]Aún no hay ejecuciones guardadas. Corre un benchmark primero.[/yellow]")
        return

    table = Table(title="Ejecuciones Guardadas en ~/.agent-tps-bench/runs/", show_header=True)
    table.add_column("#", style="dim")
    table.add_column("Fecha", style="cyan")
    table.add_column("Proveedor", style="bold")
    table.add_column("Modelo", style="magenta")
    table.add_column("Estado", style="green")
    table.add_column("Decode TPS", style="bold green")
    table.add_column("TTFT", style="yellow")
    table.add_column("Prompt", style="dim")

    for idx, r in enumerate(runs, 1):
        ttft_str = f"{r['ttft_ms']:.1f}ms" if r.get("ttft_ms") else "N/A"
        table.add_row(
            str(idx),
            r["timestamp_iso"][:19],
            r["provider"],
            r["model"][:20],
            r["status"].upper(),
            f"{r['decode_tps']:.1f} tok/s",
            ttft_str,
            r["prompt"][:35] + "…",
        )
    console.print(table)

    if not interactive:
        return

    inspect_choice = inquirer.confirm(message="¿Deseas inspeccionar una corrida en detalle?", default=False).execute()
    if inspect_choice:
        choices = [Choice(r["file_path"], f"{r['timestamp_iso'][:19]} • [{r['provider']}] {r['model']} ({r['decode_tps']:.1f} TPS)") for r in runs]
        chosen_file = inquirer.select(message="Selecciona la corrida a inspeccionar:", choices=choices).execute()
        loaded = storage.load_run(chosen_file)
        if loaded:
            res_dict = loaded.get("result", {})
            res_obj = BenchmarkResult.model_validate(res_dict)
            render_benchmark_result(res_obj)
            output_text = loaded.get("full_output") or res_obj.raw_response_preview
            if output_text:
                console.print(Panel(output_text, title="Texto Generado", border_style="cyan"))


def interactive_opencode_history() -> None:
    """Reads live OpenCode SQLite database and shows previous sessions."""
    console.print("\n[bold cyan]─── 🗄️ Historial Directo de OpenCode SQLite (opencode.db) ───[/bold cyan]")
    reader = OpenCodeDBReader()
    if not reader.is_available():
        console.print(f"[red]No se encontró la base de datos de OpenCode en {reader.db_path}[/red]")
        return
    sessions = reader.get_recent_sessions(limit=12)
    render_opencode_sessions(sessions)


def interactive_help() -> None:
    """Shows comprehensive documentation and guidance on all metrics and usage."""
    console.print("\n[bold cyan]─── ❓ Guía de Métricas, Timeouts y Comandos ───[/bold cyan]")

    help_content = """
[bold yellow]1. Métricas de TPS y Latencia:[/bold yellow]
• [bold green]Decode TPS (Tokens/seg)[/bold green]: Mide la velocidad pura de generación de la GPU:
  Fórmula: (Tokens de salida + Tokens de razonamiento) / Tiempo activo de emisión de tokens.
• [bold cyan]End-to-End TPS[/bold cyan]: Velocidad total percibida incluyendo la espera inicial (TTFT):
  Fórmula: Tokens generados / Tiempo total desde que se envía la solicitud.
• [bold magenta]TTFT (Time to First Token)[/bold magenta]: Tiempo desde el envío hasta que llega el primer token.
  Refleja la latencia de red, tiempo en cola del proveedor y el prefill del prompt.
• [bold blue]ITL (Inter-Token Latency)[/bold blue]: Intervalo promedio entre tokens durante el streaming.
• [bold white]Jitter[/bold white]: Varianza de llegada entre tokens (RFC 3550); menor jitter = streaming más fluido.

[bold yellow]2. Clasificación de Timeouts y Errores:[/bold yellow]
• [red]CONNECT_TIMEOUT[/red]: No se pudo conectar al socket o puerto del agente.
• [red]TTFT_TIMEOUT[/red]: El primer token tardó más del umbral configurado (cola saturada).
• [red]STALL_TIMEOUT[/red]: El stream se congeló a mitad de la respuesta (watchdog de latido).
• [red]DEADLINE_TIMEOUT[/red]: La ejecución completa superó el tiempo máximo permitido.
• [yellow]RATE_LIMIT_429[/yellow]: Cuota o límite por minuto alcanzado; detecta Retry-After.
• [red]PROCESS_CRASH[/red]: El proceso del CLI terminó con error (ej. SQLite bloqueado).

[bold yellow]3. Concurrencia de Subagentes:[/bold yellow]
• [bold]Degradación de Concurrencia[/bold]: Porcentaje que cae el TPS de cada trabajador individual
  al aumentar la carga de subagentes en paralelo frente a 1 trabajador en solitario.
• [bold]Aggregate Decode TPS[/bold]: Capacidad total del clúster sumando todos los subagentes.

[bold yellow]4. Comandos de Terminal para Scripts:[/bold yellow]
• [cyan]tps-bench opencode-bench -p "..."[/cyan] : Benchmark contra OpenCode CLI
• [cyan]tps-bench stream-bench --provider groq -m llama-3.3-70b-versatile[/cyan] : Benchmark API streaming
• [cyan]tps-bench stress --target opencode-server -c 4 -n 8[/cyan] : Prueba de estrés concurrente
• [cyan]tps-bench opencode-history --limit 10[/cyan] : Historial de sesiones SQLite
• [cyan]tps-bench ui[/cyan] : Abre esta interfaz interactiva
"""
    console.print(Panel(help_content, title="[bold green]Documentación del Benchmark[/bold green]", border_style="cyan"))


def run_interactive_tui() -> None:
    """Main interactive TUI loop."""
    render_banner()

    while True:
        menu_choice = inquirer.select(
            message="¿Qué deseas hacer?",
            choices=[
                Choice("single", "🚀 Ejecutar Benchmark Individual (Prompt, Modelo y Métricas)"),
                Choice("stress", "⚡ Stress Test / Concurrencia de Subagentes (Saturación y Degradación)"),
                Choice("global_search", "🔍 Buscador Global de Modelos (Búsqueda en todos los proveedores)"),
                Choice("saved_runs", "📜 Ver Historial de Ejecuciones Guardadas (~/.agent-tps-bench/runs/)"),
                Choice("opencode_db", "🗄️  Inspeccionar Base de Datos de OpenCode (opencode.db)"),
                Choice("help", "❓ Ayuda y Explicación de Métricas"),
                Choice("exit", "🚪 Salir"),
            ],
        ).execute()

        if menu_choice == "single":
            interactive_single_benchmark()
        elif menu_choice == "stress":
            interactive_stress_benchmark()
        elif menu_choice == "global_search":
            prov, mod = select_model_globally()
            console.print(f"\n[green bold]✔ Modelo seleccionado:[/green bold] [{prov.upper()}] [cyan]{mod}[/cyan]")
            run_now = inquirer.confirm(message="¿Deseas ejecutar un benchmark con este modelo ahora?", default=True).execute()
            if run_now:
                # Run with chosen model
                prompt = inquirer.text(
                    message="Ingresa el prompt a evaluar:",
                    default="Explica en 2 líneas qué es la latencia de red.",
                ).execute()
                runner = get_runner_for_provider(prov)
                with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
                    task = progress.add_task("[yellow]Generando tokens...[/yellow]", total=None)
                    result = asyncio.run(
                        runner.run_prompt(prompt=prompt, model=mod)
                        if hasattr(runner, "run_prompt")
                        else runner.run_stream(model=mod, prompt=prompt)
                    )
                render_benchmark_result(result)
                saved = storage.save_run(result, prompt)
                console.print(f"[green]✔ Guardado en {saved}[/green]")
        elif menu_choice == "saved_runs":
            interactive_view_runs()
        elif menu_choice == "opencode_db":
            interactive_opencode_history()
        elif menu_choice == "help":
            interactive_help()
        elif menu_choice == "exit":
            console.print("[cyan]¡Hasta luego![/cyan]")
            break

        console.print("\n" + "─" * 60 + "\n")
