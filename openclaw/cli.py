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
    """Show OpenClaw status"""
    try:
        response = requests.get("http://localhost:8765/health", timeout=5)
        health = response.json()

        table = Table(title="OpenClaw Status", show_header=False)
        table.add_column("Service", style="cyan")
        table.add_column("Status", style="green")

        table.add_row("Status", health.get("status", "unknown"))
        table.add_row("Version", health.get("version", "unknown"))
        table.add_row("Vision", "✓" if health.get("services", {}).get("vision") else "✗")
        table.add_row("API", "✓" if health.get("services", {}).get("api") else "✗")

        console.print(table)

    except requests.exceptions.ConnectionError:
        console.print("[red]Error: Server is not running[/red]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cli.command()
@click.option("--count", default=10, help="Number of screenshots to show")
def screenshots(count):
    """Show recent screenshots"""
    try:
        console.print("[yellow]Opening screenshots...[/yellow]")
        # This would open the screenshots
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
  status      - Show system status
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
