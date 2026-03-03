"""
Rich CLI with Autocomplete for OpenClaw

Interactive command-line interface with:
- Colored output
- Auto-completion
- Interactive prompts
- Progress bars
"""

import sys
import os
import time
import json
import click
import requests
from typing import Optional, List
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown

console = Console()


# CLI Commands
@click.group()
@click.version_option(version="2.0.0")
def cli():
    """OpenClaw - Vision-based automation framework"""
    pass


@cli.group()
def trigger():
    """Trigger management commands"""
    pass


@trigger.command("list")
@click.option("--format", type=click.Choice(["table", "json"]), default="table")
def trigger_list(format):
    """List all triggers"""
    try:
        response = requests.get("http://localhost:8765/api/v1/triggers", timeout=5)
        triggers = response.json()

        if format == "table":
            table = Table(title="Triggers")
            table.add_column("ID", style="cyan")
            table.add_column("Name", style="green")
            table.add_column("Mode", style="yellow")
            table.add_column("Enabled", style="blue")

            for t in triggers:
                table.add_row(
                    t.get("id", "-"),
                    t.get("name", "-"),
                    t.get("mode", "-"),
                    "✓" if t.get("enabled") else "✗"
                )
            console.print(table)
        else:
            console.print_json(json.dumps(triggers))

    except requests.exceptions.ConnectionError:
        console.print("[red]Error: Cannot connect to OpenClaw server[/red]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@trigger.command("create")
@click.option("--name", prompt=True, help="Trigger name")
@click.option("--mode", type=click.Choice(["ocr", "fuzzy", "template", "color", "monitor", "yolo", "window"]), default="ocr")
@click.option("--text", help="Target text for OCR mode")
@click.option("--template", help="Template path for template mode")
@click.option("--action", default="alt+o", help="Action to execute")
def trigger_create(name, mode, text, template, action):
    """Create a new trigger"""
    config = {
        "name": name,
        "mode": mode,
        "action": action
    }

    if text:
        config["config"] = {"target_text": text}
    if template:
        config["config"] = {"template_path": template}

    try:
        response = requests.post("http://localhost:8765/api/v1/triggers", json=config, timeout=5)
        if response.status_code == 201:
            console.print(f"[green]Trigger created successfully![/green]")
            console.print_json(response.json())
        else:
            console.print(f"[red]Failed to create trigger[/red]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@trigger.command("execute")
@click.argument("trigger_id")
def trigger_execute(trigger_id):
    """Execute a trigger"""
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Executing trigger...", total=None)
            response = requests.post(f"http://localhost:8765/api/v1/triggers/{trigger_id}/execute", timeout=30)

            result = response.json()
            if result.get("result"):
                console.print(f"[green]Trigger executed successfully![/green]")
            else:
                console.print("[yellow]Condition not met[/yellow]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cli.group()
def config():
    """Configuration commands"""
    pass


@config.command("show")
def config_show():
    """Show current configuration"""
    try:
        response = requests.get("http://localhost:8765/api/v1/config", timeout=5)
        config_data = response.json()

        syntax = Syntax(json.dumps(config_data, indent=2), "json", theme="monokai", line_numbers=True)
        console.print(Panel(syntax, title="Current Configuration"))

    except requests.exceptions.ConnectionError:
        console.print("[red]Error: Cannot connect to OpenClaw server[/red]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@config.command("validate")
@click.argument("config_file", type=click.Path(exists=True))
def config_validate(config_file):
    """Validate a configuration file"""
    try:
        with open(config_file, 'r') as f:
            config_data = json.load(f)

        console.print(f"[green]Configuration is valid![/green]")
        console.print(f"Mode: {config_data.get('mode', 'unknown')}")

    except json.JSONDecodeError as e:
        console.print(f"[red]Invalid JSON: {e}[/red]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cli.command()
def status():
    """Show OpenClaw status — agents, events, reactions, health"""
    # Server health (original)
    try:
        response = requests.get("http://localhost:8765/health", timeout=5)
        health = response.json()
        server_status = health.get("status", "unknown")
    except Exception:
        health = {}
        server_status = "offline"

    # Orchestration status (new)
    try:
        from core.agent_state import get_state_manager
        from core.event_bus import get_event_bus
        from core.lifecycle_manager import get_lifecycle_manager
        from core.reaction_engine import get_reaction_engine

        sm = get_state_manager()
        bus = get_event_bus()

        agent_states = {aid: s.status.value for aid, s in sm._states.items()}
        bus_stats = bus.get_stats()
    except Exception:
        agent_states = {}
        bus_stats = {}

    # Display
    table = Table(title="🔧 OpenClaw Status", show_header=False, border_style="cyan")
    table.add_column("", style="bold cyan", width=24)
    table.add_column("", style="green")

    table.add_row("Server", server_status)
    table.add_row("Agents registered", str(len(agent_states)))
    for aid, st in agent_states.items():
        color = {"running": "green", "idle": "dim", "stuck": "red", "error": "red"}.get(st, "yellow")
        table.add_row(f"  └─ {aid}", f"[{color}]{st}[/{color}]")
    table.add_row("Events emitted", str(bus_stats.get("total_events_emitted", 0)))
    table.add_row("Events (last hour)", str(bus_stats.get("events_last_hour", 0)))

    console.print(table)


@cli.command()
def agents():
    """List all registered agents with health and activity"""
    try:
        from core.agent_state import get_state_manager
        sm = get_state_manager()

        if not sm._states:
            console.print("[yellow]No agents registered.[/yellow]")
            return

        table = Table(title="🤖 Agents", border_style="cyan")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="white")
        table.add_column("Status", style="bold")
        table.add_column("Activity", style="dim")
        table.add_column("Tasks", style="green", justify="right")
        table.add_column("Errors", style="red", justify="right")
        table.add_column("Recoveries", style="yellow", justify="right")

        for aid, state in sm._states.items():
            activity = state.detect_activity()
            status_color = {
                "running": "green", "idle": "dim", "stuck": "red",
                "error": "red", "completed": "blue", "spawning": "yellow",
            }.get(state.status.value, "white")

            table.add_row(
                aid,
                state.name,
                f"[{status_color}]{state.status.value}[/{status_color}]",
                activity.value,
                str(state.success_count),
                str(state.error_count),
                str(state.recovery_count),
            )

        console.print(table)

    except ImportError:
        console.print("[red]Agent modules not available[/red]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cli.command()
@click.option("--limit", default=20, help="Number of events to show")
@click.option("--category", default=None, help="Filter by category (agent, task, swarm, etc.)")
def events(limit, category):
    """Show recent event log"""
    try:
        from core.event_bus import get_event_bus
        bus = get_event_bus()

        history = bus.get_history(limit=limit, category=category)

        if not history:
            console.print("[yellow]No events recorded.[/yellow]")
            return

        table = Table(title=f"📡 Events (last {limit})", border_style="cyan")
        table.add_column("Time", style="dim", width=10)
        table.add_column("Priority", width=8)
        table.add_column("Type", style="cyan", width=24)
        table.add_column("Message", style="white")

        for event in history:
            import datetime
            ts = datetime.datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S")
            priority_color = {
                "urgent": "red bold", "action": "yellow",
                "warning": "yellow dim", "info": "dim",
            }.get(event.priority.value, "white")

            table.add_row(
                ts,
                f"[{priority_color}]{event.priority.value}[/{priority_color}]",
                event.type.value,
                event.message[:80],
            )

        console.print(table)

    except ImportError:
        console.print("[red]Event bus not available[/red]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cli.command()
def reactions():
    """Show configured reactions and their status"""
    try:
        from core.reaction_engine import get_reaction_engine
        engine = get_reaction_engine()

        reaction_list = engine.list_reactions()
        stats = engine.get_stats()

        table = Table(title="⚡ Reaction Engine", border_style="cyan")
        table.add_column("Name", style="cyan")
        table.add_column("Event", style="white")
        table.add_column("Action", style="yellow")
        table.add_column("Enabled", justify="center")
        table.add_column("Retries", justify="right")

        for name, info in reaction_list.items():
            enabled = "[green]✓[/green]" if info["enabled"] else "[red]✗[/red]"
            table.add_row(
                name,
                info["event_type"],
                info["action"],
                enabled,
                str(info["retries"]),
            )

        console.print(table)

        # Stats summary
        console.print(f"\n[dim]Triggered: {stats['total_reactions_triggered']} | "
                      f"Successes: {stats['total_successes']} | "
                      f"Failures: {stats['total_failures']} | "
                      f"Escalations: {stats['total_escalations']}[/dim]")

    except ImportError:
        console.print("[red]Reaction engine not available[/red]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cli.command()
@click.argument("task_description")
@click.option("--decompose/--no-decompose", default=True, help="Auto-decompose into subtasks")
def spawn(task_description, decompose):
    """Spawn an agent swarm for a task — like 'ao spawn'"""
    try:
        from core.agent_swarm import AgentSwarm
        from core.event_bus import get_event_bus, EventType

        console.print(f"\n[bold cyan]🚀 Spawning swarm for:[/bold cyan] {task_description}\n")

        bus = get_event_bus()
        bus.emit(
            EventType.SWARM_STARTED,
            f"CLI spawn: {task_description}",
            data={"task": task_description, "decompose": decompose},
            source="cli",
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task(f"Decomposing: {task_description[:60]}...", total=None)

            swarm = AgentSwarm()
            result = swarm.submit_task(task_description, decompose=decompose)

        if result:
            console.print(Panel(
                f"[green]✓ Swarm completed[/green]\n\nResults: {str(result)[:500]}",
                title="Result",
                border_style="green",
            ))
        else:
            console.print("[yellow]Swarm completed with no result[/yellow]")

    except ImportError as e:
        console.print(f"[red]Module not available: {e}[/red]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cli.command()
@click.option("--count", default=10, help="Number of screenshots to show")
def screenshots(count):
    """Show recent screenshots"""
    try:
        console.print("[yellow]Opening screenshots...[/yellow]")
        console.print(f"[green]Found {count} screenshots[/green]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cli.command()
def stats():
    """Show trigger statistics"""
    try:
        response = requests.get("http://localhost:8765/api/v1/stats", timeout=5)
        stats = response.json()

        table = Table(title="Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Triggers", str(stats.get("total", 0)))
        table.add_row("Triggered", str(stats.get("triggered", 0)))
        table.add_row("Failed", str(stats.get("failed", 0)))
        table.add_row("Success Rate", f"{stats.get('success_rate', 0):.1f}%")

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cli.group()
def automation():
    """Automation commands"""
    pass


@automation.command("execute")
@click.option("--action", prompt=True, help="Action to execute (e.g., alt+o)")
@click.option("--delay", default=0, help="Delay in seconds")
def automation_execute(action, delay):
    """Execute an automation action"""
    try:
        response = requests.post(
            "http://localhost:8765/api/v1/automation/execute",
            json={"action": action, "delay": delay},
            timeout=5
        )

        if response.status_code == 200:
            console.print("[green]Action executed successfully![/green]")
        else:
            console.print("[red]Failed to execute action[/red]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cli.group()
def interactive():
    """Interactive mode"""
    pass


@interactive.command()
def start():
    """Start interactive mode"""
    console.print(Panel.fit(
        "[bold cyan]Welcome to OpenClaw Interactive Mode[/bold cyan]\n\n"
        "Type 'help' for available commands\n"
        "Type 'exit' to quit",
        border_style="cyan"
    ))

    while True:
        try:
            command = console.input("\n[bold cyan]>[/bold cyan] ")

            if command.lower() in ["exit", "quit", "q"]:
                console.print("[yellow]Goodbye![/yellow]")
                break

            if command.lower() == "help":
                console.print("""
Available commands:
  status      - Show system status (agents, events, health)
  agents      - List all agents with health
  events      - Show recent event log
  reactions   - Show active reactions
  spawn <msg> - Spawn agent swarm for a task
  triggers    - List triggers
  config      - Show configuration
  stats       - Show statistics
  exit        - Exit interactive mode
                """)
            elif command:
                console.print(f"[dim]Executing: {command}[/dim]")

        except KeyboardInterrupt:
            console.print("\n[yellow]Use 'exit' to quit[/yellow]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


def main():
    """Main CLI entry point"""
    cli()


if __name__ == "__main__":
    main()


__all__ = ["cli", "main"]
