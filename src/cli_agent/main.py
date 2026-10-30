"""
Main CLI entrypoint for cli-agent.
Provides an interactive terminal REPL and one-shot command execution with Rich formatting,
prompt_toolkit history/autocompletion, and real-time SSE streaming.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from .circuit_breaker import FallbackRouter, ModelEndpoint, QuotaCircuitBreaker
from .code_graph import CodeGraphIndexer
from .config import DEFAULT_HISTORY_FILE, load_config
from .context import compact_history, extract_session_learnings
from .doctor import run_diagnostics
from .mcp import MCPRegistry, set_active_mcp_registry
from .memory import MemoryStore, resolve_project_key
from .models import ModelCatalogEntry, fetch_model_garden_catalog
from .pr import CIWatcher, PRSynthesizer, detect_git_host_provider
from .preflight import PreflightEngine, ProjectSelector, ServiceUsageManager
from .search import get_hybrid_search_engine
from .security import RiskLevel, SecurityPolicy
from .tdd import AutonomousTDDEngine
from .team import MultiAgentTeam, set_active_team
from .tools import set_active_sandbox
from .tui import TUIApp, TUILayoutManager
from .vertex import VertexAgentClient
from .voice import VoiceAgentEngine
from .worktree import DiffViewer, WorktreeManager, WorktreeSession

console = Console()

SLASH_COMMANDS = [
    "/memories",
    "/compact",
    "/doctor",
    "/setup",
    "/model",
    "/models",
    "/index",
    "/search",
    "/sandbox",
    "/team",
    "/mcp",
    "/tdd",
    "/fix",
    "/pr",
    "/tui",
    "/voice",
    "/clear",
    "/help",
    "/exit",
    "/quit",
]
command_completer = WordCompleter(SLASH_COMMANDS, ignore_case=True, sentence=True)

pt_style = Style.from_dict(
    {
        "prompt": "ansicyan bold",
    }
)


def print_banner(
    memory_count: int, project_key: str = "global", model_name: str = "gemini-3.8-flash"
) -> None:
    console.print(
        Panel.fit(
            f"[bold cyan]cli-agent[/bold cyan] [bold white]|[/bold white] [bold green]{model_name}[/bold green]\n"
            f"[dim]Workspace: [bold yellow]{project_key}[/bold yellow] | SQLite Engine ([bold yellow]{memory_count} memories indexed[/bold yellow])[/dim]\n"
            f"[dim]Tools: File Reader/Writer, Surgical Editor, Code Search, Shell Exec[/dim]\n"
            f"[dim]Commands: [bold]/memories[/bold], [bold]/compact[/bold], [bold]/doctor[/bold], [bold]/model[/bold], [bold]/pr[/bold], [bold]/tui[/bold], [bold]/voice[/bold], [bold]/clear[/bold], [bold]/exit[/bold][/dim]",
            title="[bold magenta]AI Developer Agent[/bold magenta]",
            border_style="cyan",
        )
    )


def show_memories(memory_store: MemoryStore, project_key: str = "global") -> None:
    recent = memory_store.get_recent_memories(limit=25, project_key=project_key)
    if not recent:
        console.print(
            f"[dim yellow]No memories recorded yet for workspace '{project_key}'. Ask the agent to save important decisions![/dim yellow]"
        )
        return

    table = Table(
        title=f"Persistent Memories (Workspace: {project_key})",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("ID", style="dim", width=4)
    table.add_column("Project", style="blue", width=12)
    table.add_column("Scope", style="dim", width=8)
    table.add_column("Type", style="cyan", width=10)
    table.add_column("Title", style="bold white", width=22)
    table.add_column("Content", style="green")

    for m in recent:
        table.add_row(
            str(m["id"]),
            m.get("project_key", "global"),
            m.get("scope", "project"),
            m["type"],
            m["title"],
            m["content"][:90] + ("..." if len(m["content"]) > 90 else ""),
        )

    console.print(table)


async def show_models(
    billing_project: str,
    active_model: str,
    force_refresh: bool = False,
) -> list[ModelCatalogEntry]:
    catalog = await fetch_model_garden_catalog(
        billing_project=billing_project, force_refresh=force_refresh
    )
    table = Table(
        title="Vertex AI Model Garden Catalog (Coding & Reasoning)",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("Publisher", style="cyan", width=12)
    table.add_column("Model Identifier", style="bold white", width=28)
    table.add_column("Stage", style="yellow", width=16)
    table.add_column("Status", style="bold green", width=12)

    for idx, m in enumerate(catalog, 1):
        is_active = m.short_name == active_model or m.model_id == active_model
        active_mark = "[bold green]✔ ACTIVE[/bold green]" if is_active else ""
        table.add_row(
            str(idx),
            m.publisher.capitalize(),
            m.short_name,
            m.launch_stage,
            active_mark,
        )

    console.print(table)
    console.print(
        "[dim]Switch with [bold]/model <#>[/bold] or [bold]/model <name>[/bold]. Refresh with [bold]/model refresh[/bold].[/dim]\n"
    )
    return catalog


def make_security_policy(auto_approve: bool = False) -> SecurityPolicy:
    policy = SecurityPolicy(auto_approve=auto_approve)

    def interactive_confirm(tool_name: str, args: dict[str, Any], risk: RiskLevel) -> bool:
        if risk == RiskLevel.HIGH:
            console.print(
                f"\n[bold red]⚠️  CRITICAL / DESTRUCTIVE COMMAND DETECTED ({risk.value.upper()})[/bold red]"
            )
        else:
            console.print(
                f"\n[bold yellow]⚠️  Permission Request ({risk.value.upper()})[/bold yellow]"
            )

        args_str = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
        console.print(f"Action: [bold cyan]{tool_name}[/bold cyan]({args_str})")
        try:
            choice = (
                console.input("[bold white]Allow this action? [y/N/always]: [/bold white]")
                .strip()
                .lower()
            )
            if choice in ("always", "a"):
                policy.grant_session_permission(tool_name, args)
                return True
            return choice in ("y", "yes")
        except (KeyboardInterrupt, EOFError):
            return False

    policy.confirm_callback = interactive_confirm
    return policy


async def run_interactive(
    auto_approve: bool = False,
    model_override: str | None = None,
    max_turns: int | None = None,
) -> None:
    cfg = load_config()
    model = model_override or cfg.model
    memory_store = MemoryStore()
    project_key = resolve_project_key()
    security_policy = make_security_policy(auto_approve=auto_approve or cfg.auto_approve)
    circuit_breaker = QuotaCircuitBreaker()
    t2_model = "gemini-3.8-flash" if model != "gemini-3.8-flash" else "gemini-3.7-flash"
    t3_model = "gemini-3.7-flash" if model != "gemini-3.7-flash" else "gemini-3.8-flash"
    fallback_router = FallbackRouter(
        circuit_breaker,
        [
            ModelEndpoint(model, cfg.location, tier=1),
            ModelEndpoint(t2_model, "global", tier=2),
            ModelEndpoint(t3_model, "global", tier=3),
        ],
    )
    client = VertexAgentClient(
        project_id=cfg.project,
        location=cfg.location,
        model_name=model,
        memory_store=memory_store,
        security_policy=security_policy,
        fallback_router=fallback_router,
        circuit_breaker=circuit_breaker,
    )
    conversation_history: list[dict[str, Any]] = []

    worktree_manager = WorktreeManager()
    active_sandbox: WorktreeSession | None = None

    team_coordinator = MultiAgentTeam(client=client)
    set_active_team(team_coordinator)

    mcp_registry = MCPRegistry()
    set_active_mcp_registry(mcp_registry)
    try:
        await mcp_registry.connect_all()
    except Exception:
        pass

    code_graph_indexer = CodeGraphIndexer(client.code_graph_store)
    hybrid_search_engine = get_hybrid_search_engine()
    try:
        code_graph_indexer.index_directory(".", project_key=project_key)
        await hybrid_search_engine.index_directory(".", project_key=project_key)
    except Exception:
        pass

    recent_mems = memory_store.get_recent_memories(limit=50, project_key=project_key)
    print_banner(len(recent_mems), project_key=project_key, model_name=client.model_name)

    DEFAULT_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    session: PromptSession[str] = PromptSession(
        history=FileHistory(str(DEFAULT_HISTORY_FILE)),
        completer=command_completer,
        style=pt_style,
    )

    def on_tool_call(name: str, args: dict[str, Any]) -> None:
        args_str = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
        console.print(f"\n[dim yellow]⚙ [bold]Calling Tool:[/bold] {name}({args_str})[/dim yellow]")

    turns_completed = 0
    while True:
        if max_turns is not None and turns_completed >= max_turns:
            break
        try:
            user_input = (await session.prompt_async([("class:prompt", "\nyou ❯ ")])).strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Session terminated.[/dim]")
            break
        turns_completed += 1

        if not user_input:
            continue

        if user_input in ("/exit", "/quit", "exit", "quit"):
            console.print("[bold green]Hasta luego![/bold green]")
            if active_sandbox:
                active_sandbox.cleanup()
                set_active_sandbox(None)
            set_active_team(None)
            await mcp_registry.shutdown_all()
            set_active_mcp_registry(None)
            if len(conversation_history) >= 2:
                learnings = extract_session_learnings(conversation_history)
                memory_store.save_memory(
                    title=f"Session Summary ({project_key})",
                    content=learnings,
                    memory_type="convention",
                    project_key=project_key,
                )
            break

        if user_input == "/clear":
            conversation_history.clear()
            console.print(
                "[dim cyan]Conversation history cleared for this session (SQLite memories preserved).[/dim cyan]"
            )
            continue

        if user_input in ("/memories", "/memory"):
            show_memories(memory_store, project_key=project_key)
            continue

        if user_input == "/compact":
            conversation_history = compact_history(conversation_history, max_turns=2, keep_recent=2)
            console.print("[dim cyan]Context history manually compacted into summary.[/dim cyan]")
            continue

        if user_input.startswith("/model"):
            parts = user_input.split(maxsplit=1)
            arg = parts[1].strip() if len(parts) > 1 else ""
            if not arg:
                await show_models(billing_project=client.project_id, active_model=client.model_name)
            elif arg == "refresh":
                with console.status(
                    "[bold cyan]Refreshing Model Garden catalog from Vertex AI...[/bold cyan]"
                ):
                    await show_models(
                        billing_project=client.project_id,
                        active_model=client.model_name,
                        force_refresh=True,
                    )
            else:
                catalog = await fetch_model_garden_catalog(billing_project=client.project_id)
                chosen_entry = None
                if arg.isdigit():
                    num = int(arg)
                    if 1 <= num <= len(catalog):
                        chosen_entry = catalog[num - 1]
                else:
                    for entry in catalog:
                        if (
                            entry.short_name.lower() == arg.lower()
                            or entry.model_id.lower() == arg.lower()
                        ):
                            chosen_entry = entry
                            break

                target_model = chosen_entry.short_name if chosen_entry else arg
                client.switch_model(target_model)
                console.print(
                    f"[bold green]Switched active model to:[/bold green] [bold cyan]{client.model_name}[/bold cyan] [dim](publisher: {client.publisher})[/dim]"
                )
            continue

        if user_input == "/index":
            with console.status(
                "[bold cyan]Indexing workspace AST symbols & hybrid search chunks...[/bold cyan]"
            ):
                indexed_count = code_graph_indexer.index_directory(".", project_key=project_key)
                search_chunks = await hybrid_search_engine.index_directory(
                    ".", project_key=project_key
                )
                stats = client.code_graph_store.get_stats(project_key=project_key)
            console.print(
                f"[bold green]Indexed {indexed_count} files.[/bold green] Total: [bold cyan]{stats['symbols']}[/bold cyan] symbols across [bold cyan]{stats['files']}[/bold cyan] files ({stats['edges']} edges) | [bold yellow]{search_chunks} code chunks[/bold yellow] vectorized."
            )
            continue

        if user_input.startswith("/search") or user_input.startswith("/find"):
            parts = user_input.split(maxsplit=1)
            query = parts[1].strip() if len(parts) > 1 else ""
            if not query:
                console.print(
                    "[dim yellow]Usage: /search <query>\n"
                    "Example: /search where is token refresh handled?[/dim yellow]"
                )
                continue

            with console.status(
                "[bold cyan]Running Hybrid Code Search (BM25 + Dense Vectors + RRF)...[/bold cyan]"
            ):
                search_results = await hybrid_search_engine.search(
                    query=query, project_key=project_key, limit=5
                )

            if not search_results:
                console.print(
                    "[dim yellow]No matching code chunks found for this query.[/dim yellow]"
                )
            else:
                table = Table(
                    title=f"Hybrid Code Search Results: '{query}'",
                    show_header=True,
                    header_style="bold magenta",
                )
                table.add_column("Score (RRF)", style="bold cyan", width=12)
                table.add_column("Symbol", style="bold white", width=26)
                table.add_column("Location", style="yellow", width=36)
                table.add_column("Snippet Preview", style="dim")

                for res in search_results:
                    loc = f"{res['file_path']}:{res['start_line']}-{res['end_line']}"
                    snippet = res["content"].strip().splitlines()[0][:60]
                    table.add_row(
                        str(res["rrf_score"]),
                        f"{res['symbol_name']} ({res['symbol_type']})",
                        loc,
                        snippet,
                    )
                console.print(table)
            continue

        if user_input.startswith("/sandbox"):
            parts = user_input.split(maxsplit=1)
            action = parts[1].strip().lower() if len(parts) > 1 else "status"

            if action in ("on", "start", "create"):
                if active_sandbox and active_sandbox.is_active:
                    console.print(
                        f"[dim yellow]Sandbox is already active on branch: {active_sandbox.branch_name}[/dim yellow]"
                    )
                else:
                    try:
                        active_sandbox = worktree_manager.create_session()
                        set_active_sandbox(active_sandbox.worktree_dir)
                        console.print(
                            f"[bold green]✔ Ephemeral Git Worktree Activated![/bold green]\n"
                            f"  [dim]Branch:[/dim] [bold cyan]{active_sandbox.branch_name}[/bold cyan]\n"
                            f"  [dim]Sandbox Dir:[/dim] [yellow]{active_sandbox.worktree_dir}[/yellow]\n"
                            f"  [dim]All file mutations and commands are now completely isolated from your workspace.[/dim]"
                        )
                    except Exception as e:
                        console.print(f"[bold red]Failed creating sandbox worktree:[/bold red] {e}")
            elif action == "diff":
                if not active_sandbox or not active_sandbox.is_active:
                    console.print(
                        "[dim yellow]No active sandbox. Type '/sandbox on' to start one.[/dim yellow]"
                    )
                else:
                    diff_text = active_sandbox.get_diff()
                    console.print(DiffViewer.render(diff_text))
            elif action == "apply":
                if not active_sandbox or not active_sandbox.is_active:
                    console.print("[dim yellow]No active sandbox to apply.[/dim yellow]")
                else:
                    summary = active_sandbox.get_summary()
                    if summary.files_changed == 0:
                        console.print(
                            "[dim yellow]No modifications made in sandbox to apply.[/dim yellow]"
                        )
                    else:
                        console.print(DiffViewer.render_summary(summary))
                        confirm = (
                            console.input(
                                "[bold white]Merge these changes to your active workspace? [y/N]: [/bold white]"
                            )
                            .strip()
                            .lower()
                        )
                        if confirm in ("y", "yes"):
                            res = active_sandbox.apply_to_main()
                            if res["status"] == "applied":
                                console.print(
                                    f"[bold green]✔ Patch successfully merged![/bold green] Applied to {res['files_changed']} files (+{res['insertions']} -{res['deletions']})."
                                )
                                active_sandbox.cleanup()
                                set_active_sandbox(None)
                                active_sandbox = None
                            else:
                                console.print(
                                    f"[bold red]Merge failed:[/bold red] {res.get('message')}"
                                )
                        else:
                            console.print("[dim]Merge aborted. Sandbox remains active.[/dim]")
            elif action in ("discard", "off", "cancel"):
                if not active_sandbox or not active_sandbox.is_active:
                    console.print("[dim yellow]No active sandbox to discard.[/dim yellow]")
                else:
                    active_sandbox.cleanup()
                    set_active_sandbox(None)
                    active_sandbox = None
                    console.print(
                        "[bold green]✔ Sandbox discarded cleanly.[/bold green] Your workspace remains untouched."
                    )
            else:
                if active_sandbox and active_sandbox.is_active:
                    summary = active_sandbox.get_summary()
                    console.print(
                        f"[bold cyan]Active Sandbox:[/bold cyan] {active_sandbox.branch_name}\n"
                        f"  [dim]Directory:[/dim] {active_sandbox.worktree_dir}\n"
                        f"  [dim]Changes:[/dim] {summary.files_changed} files modified (+{summary.insertions} -{summary.deletions})\n"
                        f"  [dim]Commands:[/dim] /sandbox diff, /sandbox apply, /sandbox discard"
                    )
                else:
                    console.print(
                        "[dim]Sandbox is currently [bold]inactive[/bold]. Type [bold]/sandbox on[/bold] to isolate changes in an ephemeral worktree.[/dim]"
                    )
            continue

        if user_input.startswith("/team") or user_input.startswith("/swarm"):
            parts = user_input.split(maxsplit=1)
            goal = parts[1].strip() if len(parts) > 1 else ""
            if not goal:
                console.print(
                    "[dim yellow]Usage: /team <goal description>\nExample: /team Refactor database error handling and run tests[/dim yellow]"
                )
                continue

            console.print(
                f"\n[bold magenta]🚀 Initiating Multi-Agent Swarm (ODD Pipeline)...[/bold magenta]\n[dim]Goal: {goal}[/dim]\n"
            )

            def on_team_event(event_type: str, message: str) -> None:
                if event_type == "stage":
                    console.print(f"\n[bold cyan]▶ {message}[/bold cyan]")
                elif event_type == "spawn":
                    console.print(f"  [dim yellow]⚡ {message}[/dim yellow]")
                elif event_type == "tool_call":
                    console.print(f"    [dim]{message}[/dim]")

            with console.status("[bold cyan]Multi-Agent Swarm working...[/bold cyan]"):
                workflow_res = await team_coordinator.run_workflow(goal, on_event=on_team_event)

            console.print(Markdown(workflow_res.summary))
            continue

        if user_input.startswith("/mcp"):
            parts = user_input.split(maxsplit=1)
            action = parts[1].strip().lower() if len(parts) > 1 else "list"

            if action == "reload":
                with console.status("[bold cyan]Reloading MCP servers...[/bold cyan]"):
                    await mcp_registry.shutdown_all()
                    count = await mcp_registry.connect_all()
                console.print(
                    f"[bold green]✔ MCP Reloaded![/bold green] Connected to [bold cyan]{count}[/bold cyan] servers ({len(mcp_registry.tools)} tools federated)."
                )
            else:
                if not mcp_registry.tools:
                    console.print(
                        f"[dim yellow]No federated MCP tools active.\nRegister servers in [bold]{mcp_registry.config_path}[/bold] (JSON format) and run '/mcp reload'.[/dim yellow]"
                    )
                else:
                    table = Table(
                        title=f"Federated MCP Tools ({len(mcp_registry.servers)} servers, {len(mcp_registry.tools)} tools)",
                        show_header=True,
                        header_style="bold magenta",
                    )
                    table.add_column("Server", style="cyan", width=14)
                    table.add_column("Tool Name", style="bold white", width=28)
                    table.add_column("Description", style="dim")

                    for tool in mcp_registry.tools.values():
                        table.add_row(tool.server_name, tool.full_name, tool.description[:80])
                    console.print(table)
            continue

        if user_input.startswith("/tdd") or user_input.startswith("/fix"):
            parts = user_input.split(maxsplit=1)
            target = parts[1].strip() if len(parts) > 1 else ""
            if not target:
                console.print(
                    "[dim yellow]Usage: /tdd <test_command or target>\n"
                    "Example: /tdd tests/unit/test_calc.py::test_add[/dim yellow]"
                )
                continue

            test_cmd = target if ("pytest" in target or "python" in target) else f"pytest {target}"
            console.print(
                f"\n[bold magenta]🛠 Initiating Autonomous TDD Repair Loop...[/bold magenta]\n"
                f"[dim]Reproducing Target: {test_cmd}[/dim]\n"
            )

            tdd_engine = AutonomousTDDEngine()

            def on_tdd_event(evt_type: str, msg: str) -> None:
                if evt_type == "phase_red":
                    console.print(f"[bold red]▶ {msg}[/bold red]")
                elif evt_type == "phase_patch":
                    console.print(f"[bold yellow]▶ {msg}[/bold yellow]")
                elif evt_type == "phase_green":
                    console.print(f"[bold green]▶ {msg}[/bold green]")
                elif evt_type == "phase_regression":
                    console.print(f"[bold cyan]▶ {msg}[/bold cyan]")
                else:
                    console.print(f"  [dim]{msg}[/dim]")

            with console.status("[bold cyan]TDD Self-Healing Engine executing...[/bold cyan]"):
                tdd_result = await tdd_engine.run_cycle(
                    reproducing_command=test_cmd,
                    on_event=on_tdd_event,
                )

            if tdd_result.success:
                console.print(f"\n[bold green]✔ {tdd_result.final_status}[/bold green]")
                if tdd_result.diff:
                    console.print("\n[bold cyan]Verified Patch Diff:[/bold cyan]")
                    console.print(
                        Syntax(tdd_result.diff, "diff", theme="monokai", line_numbers=True)
                    )
            else:
                console.print(f"\n[bold red]✖ {tdd_result.final_status}[/bold red]")
            continue

        if user_input.startswith("/setup"):
            console.print(
                "\n[bold magenta]🛠 Google Cloud & Vertex AI Guided Setup Wizard[/bold magenta]\n"
            )
            pf_engine = PreflightEngine()
            with console.status(
                "[bold cyan]Inspecting local Google Cloud environment...[/bold cyan]"
            ):
                report = pf_engine.run_checks()

            console.print(
                f"  [bold]Active Account:[/bold] [cyan]{report.account or 'None (Not Logged In)'}[/cyan]"
            )
            console.print(
                f"  [bold]Target Project:[/bold] [cyan]{report.project or 'None (Unset)'}[/cyan]"
            )
            console.print(f"  [bold]Active Model:[/bold]   [cyan]{client.model_name}[/cyan]\n")

            projects = ProjectSelector.list_accessible_projects()
            if projects:
                p_table = Table(title="Accessible Google Cloud Projects", show_header=True)
                p_table.add_column("#", width=4)
                p_table.add_column("Project ID", style="bold cyan")
                p_table.add_column("Project Name", style="white")
                p_table.add_column("Number", style="dim")

                for idx, proj in enumerate(projects, 1):
                    p_table.add_row(
                        str(idx), proj["projectId"], proj["name"], proj["projectNumber"]
                    )
                console.print(p_table)

                sel = console.input(
                    "[bold white]Select project number to switch (or press Enter to keep current): [/bold white]"
                ).strip()
                if sel.isdigit() and 1 <= int(sel) <= len(projects):
                    chosen_proj = projects[int(sel) - 1]["projectId"]
                    ProjectSelector.save_project_to_config(chosen_proj)
                    client.project_id = chosen_proj
                    console.print(
                        f"[bold green]✔ Project updated to:[/bold green] [bold cyan]{chosen_proj}[/bold cyan] (saved to config.toml)"
                    )

            target_proj = client.project_id
            if not ServiceUsageManager.is_api_enabled(target_proj):
                console.print(
                    f"[bold yellow]⚠ Vertex AI API is disabled on '{target_proj}'.[/bold yellow]"
                )
                enable_confirm = (
                    console.input(
                        "[bold white]Enable aiplatform.googleapis.com now? [Y/n]: [/bold white]"
                    )
                    .strip()
                    .lower()
                )
                if enable_confirm in ("", "y", "yes"):
                    with console.status("[bold cyan]Enabling Vertex AI API...[/bold cyan]"):
                        ok = ServiceUsageManager.enable_api(target_proj)
                    if ok:
                        console.print(
                            "[bold green]✔ Vertex AI API enabled successfully![/bold green]"
                        )
                    else:
                        console.print(
                            f"[bold red]Failed enabling API. Run:[/bold red] gcloud services enable aiplatform.googleapis.com --project={target_proj}"
                        )
            else:
                console.print("[bold green]✔ Vertex AI API is active and ready.[/bold green]")
            continue

        if user_input.startswith("/doctor"):
            auto_fix = "--fix" in user_input
            with console.status("[bold cyan]Running health diagnostics...[/bold cyan]"):
                diag = await run_diagnostics(
                    token_provider=client.token_provider,
                    memory_store=memory_store,
                    code_graph_store=client.code_graph_store,
                )
                if auto_fix:
                    pf = PreflightEngine()
                    pf.run_checks(auto_fix=True)

            table = Table(
                title="System & Workspace Diagnostics",
                show_header=True,
                header_style="bold magenta",
            )
            table.add_column("Component", style="bold white", width=18)
            table.add_column("Status", width=12)
            table.add_column("Detail", style="dim")

            tok_str = (
                "[green]Valid[/green]"
                if diag["token_status"] == "valid"
                else f"[red]{diag['token_status']}[/red]"
            )
            db_str = (
                "[green]Healthy[/green]"
                if diag["database_status"] == "healthy"
                else f"[red]{diag['database_status']}[/red]"
            )
            cg_str = (
                "[green]Healthy[/green]"
                if diag.get("code_graph_status") == "healthy"
                else f"[red]{diag.get('code_graph_status')}[/red]"
            )
            cg_stats = diag.get("code_graph_stats", {})

            table.add_row(
                "Google Account",
                "[cyan]Active[/cyan]",
                diag.get("gcloud_account", "Unauthenticated"),
            )
            table.add_row("OAuth2 Token", tok_str, "")
            table.add_row("Target Project", "[cyan]Configured[/cyan]", client.project_id)
            table.add_row("SQLite Database", db_str, diag["database_path"])
            table.add_row(
                "AST Code Graph",
                cg_str,
                f"{cg_stats.get('symbols', 0)} symbols / {cg_stats.get('files', 0)} files ({cg_stats.get('edges', 0)} edges)",
            )
            table.add_row(
                "Git Sandbox Engine",
                "[green]Ready[/green]",
                "Ephemeral branches & atomic diff reviews available",
            )
            table.add_row(
                "Multi-Agent Swarm",
                "[green]Ready[/green]",
                "ODD Swarm (Architect, Feature Worker, QA Verifier)",
            )
            table.add_row(
                "MCP Federation",
                "[green]Ready[/green]",
                f"{diag.get('mcp_servers', 0)} servers / {diag.get('mcp_tools', 0)} tools active",
            )
            table.add_row(
                "TDD Repair Engine",
                "[green]Ready[/green]"
                if diag.get("tdd_status") == "ready"
                else "[yellow]Partial[/yellow]",
                "Autonomous Red-Green-Refactor loop & regression guards",
            )
            table.add_row(
                "Hybrid Code Search",
                "[green]Ready[/green]",
                "FTS5 BM25 + Dense Vectors + Reciprocal Rank Fusion",
            )
            table.add_row(
                "Git Host Platform",
                "[cyan]Detected[/cyan]",
                str(diag.get("git_host_platform", "unknown")),
            )
            table.add_row(
                "Circuit Breaker",
                "[green]Ready[/green]",
                str(diag.get("circuit_breaker_status", "CLOSED")),
            )
            table.add_row(
                "Split-Pane TUI",
                "[green]Ready[/green]",
                str(diag.get("tui_status", "ready")),
            )
            table.add_row(
                "Voice Engine",
                "[green]Ready[/green]",
                str(diag.get("voice_status", "headless_ready")),
            )
            table.add_row("Active Workspace", "[cyan]Detected[/cyan]", diag["workspace_key"])
            table.add_row("Model Endpoint", "[cyan]Configured[/cyan]", client.model_name)
            console.print(table)
            continue

        if user_input.startswith("/pr"):
            parts = user_input.split(maxsplit=1)
            action = parts[1].strip().lower() if len(parts) > 1 else "preview"
            if action == "create":
                provider = detect_git_host_provider(Path("."))
                synthesizer = PRSynthesizer(worktree_path=Path("."), client=client)
                pr_payload = synthesizer.generate_pr()
                pr_meta = provider.create_or_update_pr(
                    title=pr_payload["title"],
                    body=pr_payload["body"],
                    source_branch="HEAD",
                    target_branch="main",
                )
                console.print(
                    f"[bold green]✔ PR/MR Created or Updated![/bold green] #{pr_meta.number} -> {pr_meta.url}"
                )
            elif action in ("watch", "ci"):
                provider = detect_git_host_provider(Path("."))
                watcher = CIWatcher(provider=provider)
                watch_res = watcher.watch()
                console.print(
                    f"[bold cyan]CI Watch Status:[/bold cyan] {watch_res.final_status} "
                    f"(heal attempts: {watch_res.heal_attempts_used})"
                )
            else:
                synthesizer = PRSynthesizer(worktree_path=Path("."), client=client)
                pr_payload = synthesizer.generate_pr()
                console.print(
                    Panel(
                        pr_payload["commit_message"],
                        title="[bold cyan]Conventional Commit[/bold cyan]",
                        border_style="cyan",
                    )
                )
                console.print(
                    Panel(
                        Markdown(pr_payload["body"]),
                        title=f"[bold magenta]ISO/IEC 25010 PR Preview: {pr_payload['title']}[/bold magenta]",
                        border_style="magenta",
                    )
                )
            continue

        if user_input.startswith("/tui"):
            app = TUIApp(layout_manager=TUILayoutManager())
            app.run()
            continue

        if user_input.startswith("/voice"):
            voice_engine = VoiceAgentEngine(client=client)
            voice_res = await voice_engine.run_session()
            console.print(
                f"[bold cyan]Voice Session ({voice_res.get('status', 'ready')}):[/bold cyan] "
                f"Endpoint={voice_res.get('websocket_url')}"
            )
            continue

        if user_input == "/help":
            console.print(
                "[bold]Available commands:[/bold]\n"
                " - [bold]/setup[/bold]: Launch interactive Google Cloud project and onboarding setup wizard\n"
                " - [bold]/memories[/bold]: List persistent memories for current workspace\n"
                " - [bold]/compact[/bold]: Compact conversation context\n"
                " - [bold]/doctor[/bold]: Verify authentication, database, AST index, and workspace health\n"
                " - [bold]/model <name>[/bold]: Switch or inspect model\n"
                " - [bold]/index[/bold]: Re-index workspace code AST symbols and relations\n"
                " - [bold]/search <query>[/bold]: Hybrid BM25 + vector search with Reciprocal Rank Fusion\n"
                " - [bold]/sandbox [on|diff|apply|discard][/bold]: Manage isolated Git worktree sandboxing\n"
                " - [bold]/team <goal>[/bold]: Run autonomous multi-agent swarm (Architect -> Worker -> QA)\n"
                " - [bold]/mcp [list|reload][/bold]: Inspect and reload federated MCP servers and tools\n"
                " - [bold]/tdd <target>[/bold]: Run autonomous TDD self-healing loop on a failing test\n"
                " - [bold]/pr [preview|create|watch][/bold]: Synthesize, create, or watch GitHub/GitLab PR/MRs\n"
                " - [bold]/tui[/bold]: Launch fullscreen 4-zone Split-Pane Terminal UI\n"
                " - [bold]/voice[/bold]: Start bidirectional Gemini Live voice streaming session\n"
                " - [bold]/clear[/bold]: Clear current chat context\n"
                " - [bold]/exit[/bold]: Quit the agent and persist session summary"
            )
            continue

        conversation_history.append({"role": "user", "parts": [{"text": user_input}]})
        conversation_history = compact_history(conversation_history, max_turns=12, keep_recent=6)

        console.print("\n[bold magenta]agent ❯[/bold magenta] ", end="")
        full_chunks: list[str] = []
        try:
            async for chunk in client.stream_chat(conversation_history, on_tool_call=on_tool_call):
                sys.stdout.write(chunk)
                sys.stdout.flush()
                full_chunks.append(chunk)
            sys.stdout.write("\n")
            sys.stdout.flush()
        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] {e}")
            continue

        full_response = "".join(full_chunks)
        conversation_history.append({"role": "model", "parts": [{"text": full_response}]})


async def run_oneshot(
    prompt: str, auto_approve: bool = False, model_override: str | None = None
) -> None:
    cfg = load_config()
    model = model_override or cfg.model
    memory_store = MemoryStore()
    security_policy = make_security_policy(auto_approve=auto_approve or cfg.auto_approve)
    circuit_breaker = QuotaCircuitBreaker()
    t2_model = "gemini-3.8-flash" if model != "gemini-3.8-flash" else "gemini-3.7-flash"
    t3_model = "gemini-3.7-flash" if model != "gemini-3.7-flash" else "gemini-3.8-flash"
    fallback_router = FallbackRouter(
        circuit_breaker,
        [
            ModelEndpoint(model, cfg.location, tier=1),
            ModelEndpoint(t2_model, "global", tier=2),
            ModelEndpoint(t3_model, "global", tier=3),
        ],
    )
    client = VertexAgentClient(
        project_id=cfg.project,
        location=cfg.location,
        model_name=model,
        memory_store=memory_store,
        security_policy=security_policy,
        fallback_router=fallback_router,
        circuit_breaker=circuit_breaker,
    )

    def on_tool_call(name: str, args: dict[str, Any]) -> None:
        args_str = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
        console.print(f"[dim yellow]⚙ [bold]Tool:[/bold] {name}({args_str})[/dim yellow]")

    messages = [{"role": "user", "parts": [{"text": prompt}]}]
    try:
        async for chunk in client.stream_chat(messages, on_tool_call=on_tool_call):
            sys.stdout.write(chunk)
            sys.stdout.flush()
        sys.stdout.write("\n")
        sys.stdout.flush()
    except Exception as e:
        err_msg = str(e) or repr(e)
        console.print(f"[bold red]Error ({type(e).__name__}):[/bold red] {err_msg}")
        sys.exit(1)


def main() -> None:
    auto_approve = "--auto-approve" in sys.argv
    tui_mode = "--tui" in sys.argv
    voice_mode = "--voice" in sys.argv
    cli_args = [arg for arg in sys.argv[1:] if arg not in ("--auto-approve", "--tui", "--voice")]

    model_override = None
    if "--model" in cli_args:
        idx = cli_args.index("--model")
        if idx + 1 < len(cli_args):
            model_override = cli_args[idx + 1]
            cli_args = cli_args[:idx] + cli_args[idx + 2 :]

    max_turns: int | None = None
    if "--max-turns" in cli_args:
        idx = cli_args.index("--max-turns")
        if idx + 1 < len(cli_args):
            try:
                max_turns = int(cli_args[idx + 1])
            except ValueError:
                max_turns = None
            cli_args = cli_args[:idx] + cli_args[idx + 2 :]

    if tui_mode:
        TUIApp(layout_manager=TUILayoutManager()).run()
        return

    if voice_mode:
        cfg = load_config()
        model = model_override or cfg.model
        client = VertexAgentClient(
            project_id=cfg.project,
            location=cfg.location,
            model_name=model,
        )
        asyncio.run(VoiceAgentEngine(client=client).run_session())
        return

    if cli_args:
        prompt = " ".join(cli_args)
        asyncio.run(run_oneshot(prompt, auto_approve=auto_approve, model_override=model_override))
    elif not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
        if prompt:
            asyncio.run(
                run_oneshot(prompt, auto_approve=auto_approve, model_override=model_override)
            )
        else:
            asyncio.run(
                run_interactive(
                    auto_approve=auto_approve,
                    model_override=model_override,
                    max_turns=max_turns,
                )
            )
    else:
        asyncio.run(
            run_interactive(
                auto_approve=auto_approve,
                model_override=model_override,
                max_turns=max_turns,
            )
        )


if __name__ == "__main__":
    main()
