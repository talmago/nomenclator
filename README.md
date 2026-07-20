# Nomenclator

**Nomenclator** is a Python library for Harmonized System (HS) classification using hybrid retrieval and LLM reasoning.

The project combines deterministic retrieval over the official HS nomenclature with DSPy-based reasoning modules to produce legally grounded HS classification candidates.

## Features

* 📚 Parses the official HS nomenclature into a structured tree
* 🔎 Hybrid semantic + BM25 retrieval over HS chapters
* 🤖 DSPy-based multi-stage classification pipeline
* ⚖️ Structured legal reasoning over chapter notes and headings
* 🧩 Modular architecture with independently testable components

## Installation

```bash
pip install nomenclator
```

Configure an OpenAI-compatible API key before running DSPy.

For example:

```bash
export OPENAI_API_KEY=...
```

## Quick Start

```python
import dspy

from nomenclator import HSClassificationAgent, calc_usage

# Configure DSPy
lm = dspy.LM("openai/gpt-4.1-mini")
dspy.configure(lm=lm)

def main() -> None:
    """Run a simple HS classification example."""

    agent = HSClassificationAgent()

    queries = [
        "Men's cotton knitted shirts",
        "Lithium ion battery for electric vehicles",
        "Fresh bananas",
    ]

    result = agent.classify(queries[1])

    print("Top classification candidates:\n")

    for i, candidate in enumerate(result.candidates, start=1):
        print(f"{i}. {candidate.code} — {candidate.description}")
        print(f"   Score: {candidate.score:.2f}")
        print(f"   Source chapter: {candidate.source_chapter}")
        print("   Reasoning:")

        for reason in candidate.reasoning:
            print(f"     • {reason}")

        print()

    usage = calc_usage(lm.history)

    print("Token usage")
    print(f"  Prompt tokens:     {usage.prompt_tokens:,}")
    print(f"  Completion tokens: {usage.completion_tokens:,}")
    print(f"  Total tokens:      {usage.total_tokens:,}")
    print(f"  Estimated cost:    ${usage.cost:.5f}")


if __name__ == "__main__":
    main()
```

Example output:

```text
Top classification candidates:

1. 8507.60 — Electric accumulators, including separators therefor, whether or not rectangular (including square) — Lithium-ion
   Score: 1.00
   Source chapter: 8585-2022E
   Reasoning:
     • Product is a lithium-ion rechargeable battery, an electric accumulator.
     • Heading 85.07 covers electric accumulators; 8507.60 specifically covers lithium-ion accumulators.
     • No other heading offers a more specific or appropriate classification.
     • GIR 3(a) favors the most specific applicable heading, which is 8507.60.

Token usage
  Prompt tokens:     15,406
  Completion tokens: 530
  Total tokens:      15,936
  Estimated cost:    $0.00701
```

## How it works

```
Q: "Men's cotton knitted shirts"
        │
        ▼
┌─ Product Analyst ──────────────────────────────────────────────┐
│  normalized: men's cotton knitted shirts                      │
│  category:   textile apparel                                  │
│  attrs:      type=shirt · material=cotton · knit              │
│  keywords:                                                    │
│    • cotton shirt                                             │
│    • knitted shirt                                            │
│    • knitted apparel                                          │
│    • textile apparel                                          │
│    • garments                                                 │
└────────────────────────────┬───────────────────────────────────┘
                             │
                             ▼
┌─ Chapter Retriever (1st retrieval) ────────────────────────────┐
│  hybrid search over HS chapters                               │
│                                                               │
│    Ch.61  Articles of apparel, knitted or crocheted           │
│    Ch.62  Articles of apparel, not knitted or crocheted       │
│    …                                                          │
└────────────────────────────┬───────────────────────────────────┘
                             │
                             ▼
┌─ Research Analyst ─────────────────────────────────────────────┐
│  analyze candidate chapters                                   │
│                                                               │
│    1. Ch.61  Primary pathway                                  │
│    2. Ch.62  Alternative pathway                              │
└────────────────────────────┬───────────────────────────────────┘
                             │
                             ▼
┌─ Heading Retriever (2nd retrieval) ────────────────────────────┐
│  hybrid search over heading chunks                            │
│                                                               │
│    Ch.61  61.05 → 6105.10                                     │
│    Ch.62  62.05 → 6205.20                                     │
│    chapter notes + heading context                            │
└────────────────────────────┬───────────────────────────────────┘
                             │
                 ┌───────────┴───────────┐
                 │  GIR Rules (fixed)    │
                 └───────────┬───────────┘
                             ▼
┌─ Classification Analyst ───────────────────────────────────────┐
│  rank HS classification candidates                            │
│                                                               │
│  1. 6105.10  Men's or boys' shirts, knitted, of cotton        │
│     score 1.00                                                │
│                                                               │
│  2. 6205.20  Men's or boys' shirts, of cotton                 │
│     score 0.30                                                │
└────────────────────────────┬───────────────────────────────────┘
                             │
                             ▼
┌─ Classification Result ────────────────────────────────────────┐
│  ✓ Selected: 6105.10                                          │
│                                                               │
│  Confidence: High                                             │
│  Alternatives: 6205.20                                        │
│  Reasoning:                                                    │
│    • Product is a knitted cotton shirt.                       │
│    • Chapter 61 is more specific than Chapter 62.             │
│    • GIR 3(a) favors the most specific heading.               │
└───────────────────────────────────────────────────────────────┘
```

## Development

Install the project dependencies:

```bash
poetry install
```

Optionally, install **Poe the Poet** globally for shorter commands:

```bash
pipx install poethepoet
```

Common development tasks:

```bash
poe format       # Format and auto-fix the source code
poe lint         # Run static analysis
poe typecheck    # Run type checking
poe test         # Execute the unit test suite
poe integration  # Execute the integration test suite
poe check        # Run linting, type checking, and unit tests
```

> Coding agents working in this repository follow the rules described in [`AGENTS.md`](./AGENTS.md)

## Architecture

For a detailed overview of the system design, pipeline flow, retrieval strategy,
and agent responsibilities, see:

- 👉 [Architecture](ARCHITECTURE.md)

## Roadmap

Planned improvements include:

* Heading- and subheading-level retrieval
* DSPy optimization using evaluation datasets
* Additional embedding model support
* Alternative retrieval backends
* Classification benchmarking and evaluation
* Optional audit stage for reviewing classification results

## License

MIT License.
