"""
Command-line interface for Nomenclator.

This module provides a Rich-based CLI for classifying products using the
Harmonized System (HS) nomenclature. It configures a DSPy language model,
invokes the HS classification pipeline, and renders the results in a
human-friendly terminal interface.

Examples:
    $ nomenclator "Fresh bananas"

    $ echo "Fresh bananas" | nomenclator

    $ nomenclator --model anthropic/claude-sonnet-4-5 "Electric motor"
"""

import argparse
from collections.abc import Sequence
import sys

import dspy
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from nomenclator.agent import HSClassificationAgent
from nomenclator.usage import calc_usage

DEFAULT_MODEL = "openai/gpt-4.1-mini"
DEFAULT_TOP = 3


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Args:
        argv:
            Optional command-line arguments. When ``None``, arguments are
            read from ``sys.argv``.

    Returns:
        Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(
        prog="nomenclator",
        description=("Classify products using the Harmonized System nomenclature."),
        epilog=(
            "Examples:\n"
            '  nomenclator "Fresh bananas"\n'
            '  nomenclator --top 5 "Stainless steel screws"\n'
            '  nomenclator --model openai/gpt-4.1-mini "Electric motor"\n'
            '  echo "Fresh bananas" | nomenclator'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "description",
        nargs="*",
        metavar="DESCRIPTION",
        help=(
            "Product description to classify. When omitted, the description "
            "is read from standard input."
        ),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        metavar="MODEL",
        help=(
            f"DSPy language model in provider/model format (default: {DEFAULT_MODEL})."
        ),
    )
    parser.add_argument(
        "--top",
        type=int,
        default=DEFAULT_TOP,
        metavar="N",
        help=(
            "Maximum number of classification candidates to display "
            f"(default: {DEFAULT_TOP})."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Display additional classification pipeline details.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Write the classification result as JSON.",
    )

    args = parser.parse_args(argv)

    if args.top < 1:
        parser.error("--top must be greater than zero")

    return args


def _configure_model(model: str) -> dspy.LM:
    """
    Configure DSPy with the requested language model.

    Args:
        model:
            Language model identifier in ``provider/model`` format.

    Returns:
        Configured DSPy language model.
    """

    lm = dspy.LM(model)
    dspy.configure(lm=lm)

    return lm


def _resolve_description(parts: Sequence[str]) -> str | None:
    """
    Resolve the product description from the command line or standard input.

    Command-line arguments take precedence. When no positional arguments
    are provided, the description is read from stdin.

    Args:
        parts:
            Positional command-line arguments.

    Returns:
        Product description or ``None`` if no input was provided.
    """

    if parts:
        description = " ".join(parts).strip()
        return description or None

    if not sys.stdin.isatty():
        description = sys.stdin.read().strip()
        return description or None

    return None


def render_header(
    console: Console,
    description: str,
    model: str,
) -> None:
    """
    Render the CLI header.

    Args:
        console:
            Rich console used for rendering.
        description:
            Product description being classified.
        model:
            Language model used for the classification.
    """

    title = Text("Nomenclator", style="bold cyan")
    title.append(f"  ({model})", style="dim")

    console.print(
        Panel(
            title,
            border_style="cyan",
            box=box.ROUNDED,
        )
    )

    console.print("[bold]Product[/bold]")
    console.print(f"  {description}")
    console.print()


def render_best_match(
    console: Console,
    candidate,
) -> None:
    """
    Render the highest-ranked classification candidate.

    Args:
        console:
            Rich console used for rendering.
        candidate:
            Highest-ranked classification candidate.
    """

    table = Table.grid(padding=(0, 2))
    table.add_column(style="cyan", width=12)
    table.add_column()

    table.add_row("HS Code", candidate.code)
    table.add_row("Confidence", f"{candidate.score:.2f}")
    table.add_row("Description", candidate.description)

    if getattr(candidate, "source_chapter", None):
        table.add_row("Chapter", candidate.source_chapter)

    if getattr(candidate, "source_url", None):
        table.add_row(
            "Source",
            f"[link={candidate.source_url}]{candidate.source_url}[/link]",
        )

    console.print(
        Panel(
            table,
            title="Best Classification",
            border_style="green",
            box=box.ROUNDED,
        )
    )
    console.print()


def render_alternatives(
    console: Console,
    candidates,
) -> None:
    """
    Render the ranked list of alternative classification candidates.

    The highest-ranked candidate is assumed to be displayed separately by
    :func:`render_best_match`.

    Args:
        console:
            Rich console used for rendering.
        candidates:
            Ranked classification candidates.
    """

    alternatives = list(candidates)[1:]

    if not alternatives:
        return

    table = Table(
        title="Alternative Classifications",
        box=box.SIMPLE_HEAVY,
    )

    table.add_column("Rank", justify="right", style="dim")
    table.add_column("HS Code", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Description")

    for rank, candidate in enumerate(alternatives, start=2):
        table.add_row(
            str(rank),
            candidate.code,
            f"{candidate.score:.2f}",
            candidate.description,
        )

    console.print(table)
    console.print()


def render_usage(
    console: Console,
    usage,
) -> None:
    """
    Render language model token usage and estimated cost.
    """

    table = Table.grid(padding=(0, 2))
    table.add_column(style="cyan")
    table.add_column(justify="right")

    table.add_row("Prompt tokens", f"{usage.prompt_tokens:,}")
    table.add_row("Completion tokens", f"{usage.completion_tokens:,}")
    table.add_row("Total tokens", f"{usage.total_tokens:,}")
    table.add_row("Estimated cost", f"${usage.cost:.5f}")

    console.print(
        Panel(
            table,
            title="Performance",
            box=box.ROUNDED,
        )
    )
    console.print()


def render_footer(console: Console) -> None:
    """
    Render the CLI footer.

    Args:
        console:
            Rich console used for rendering.
    """

    console.rule(style="dim")


def render_reasoning(
    console: Console,
    candidate,
) -> None:
    """
    Render the reasoning for the selected classification.

    Args:
        console:
            Rich console used for rendering.
        candidate:
            Selected classification candidate.
    """

    reasoning = getattr(candidate, "reasoning", None)

    if not reasoning:
        return

    tree = Tree("[bold]Reasoning[/bold]")

    for item in reasoning:
        tree.add(item)

    console.print(tree)
    console.print()


def render_pipeline(
    console: Console,
    result,
) -> None:
    """
    Render optional pipeline information.

    This function is only used when ``--verbose`` is enabled.

    Args:
        console:
            Rich console used for rendering.
        result:
            Classification result.
    """

    sections = [
        ("Product Facts", "product_facts"),
        ("Candidate Chapters", "chapters"),
        ("Heading Context", "heading_context"),
        ("Legal Notes", "notes"),
        ("General Rules of Interpretation", "gir"),
        ("Pipeline Reasoning", "reasoning"),
    ]

    rendered = False

    for title, attribute in sections:
        value = getattr(result, attribute, None)

        if value is None:
            continue

        if isinstance(value, str):
            body = value
        elif hasattr(value, "model_dump_json"):
            body = value.model_dump_json(indent=2)
        elif isinstance(value, list):
            body = "\n".join(f"• {item}" for item in value)
        else:
            body = str(value)

        console.print(
            Panel(
                body,
                title=title,
                box=box.ROUNDED,
            )
        )

        rendered = True

    if rendered:
        console.print()


def _classify(
    agent: HSClassificationAgent,
    console: Console,
    description: str,
):
    """
    Run product classification while displaying a progress spinner.

    Args:
        agent:
            Configured HS classification agent.
        console:
            Rich console.
        description:
            Product description.

    Returns:
        Classification result.
    """

    with console.status(
        "[bold green]Classifying product...[/]",
        spinner="dots",
    ):
        return agent.classify(description)


def main(argv: Sequence[str] | None = None) -> int:
    """
    Execute the Nomenclator command-line interface.

    Args:
        argv:
            Optional command-line arguments. When ``None``, arguments are
            read from ``sys.argv``.

    Returns:
        POSIX-compatible process exit status.
    """

    args = _parse_args(argv)

    console = Console()

    try:
        lm = _configure_model(args.model)
    except Exception as exc:
        console.print(f"[red]Failed to configure model '{args.model}':[/] {exc}")
        return 2

    description = _resolve_description(args.description)

    if description is None:
        console.print(
            "[red]No product description provided.[/]\n\n"
            "Provide a description as an argument:\n"
            '  nomenclator "Fresh bananas"\n\n'
            "or pipe it from standard input:\n"
            '  echo "Fresh bananas" | nomenclator'
        )
        return 2

    agent = HSClassificationAgent()

    try:
        result = _classify(
            agent=agent,
            console=console,
            description=description,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Classification cancelled.[/]")
        return 130
    except Exception as exc:
        console.print(f"[red]Classification failed:[/] {exc}")
        return 1

    usage = calc_usage(lm.history)

    render_header(console, description, lm.model)

    if result.candidates:
        render_best_match(console, result.candidates[0])
        render_alternatives(console, result.candidates[: args.top])
        render_reasoning(console, result.candidates[0])
    else:
        console.print("[yellow]No classification candidates returned.[/]")
        console.print()

    if args.verbose:
        render_pipeline(console, result)

    render_usage(console, usage)
    render_footer(console)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
