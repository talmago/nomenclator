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


def main() -> None:
    """Run a simple HS classification example."""

    # Configure DSPy
    lm = dspy.LM("openai/gpt-4.1-mini")
    dspy.configure(lm=lm)

    # Initialize the classifier
    agent = HSClassificationAgent()

    queries = [
        "Men's cotton knitted shirts",
        "Lithium ion battery pack for electric vehicles",
        "Fresh bananas",
    ]

    result = agent.classify(queries[0])

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

1. 6105.10 — Men's or boys' shirts, knitted or crocheted, of cotton
   Score: 1.00
   Source chapter: 61
   Reasoning:
     • Product is men's knitted shirts made of cotton, matching heading 61.05 for men's knitted shirts.
     • Subheading 6105.10 specifies cotton material which corresponds exactly to the product's material.
     • Chapter 61 applies only to made up knitted or crocheted articles per Chapter Note 1.
     • Chapter 62 covers non-knitted apparel and is therefore not applicable.
     • GIR Rules 1 and 3 favor the most specific accurate classification.
     • No exclusion or other heading conflicts were identified.

Token usage
  Prompt tokens:     6,680
  Completion tokens: 589
  Total tokens:      7,269
  Estimated cost:    $0.00924
```

## How it works?

```
Q: "Men's cotton knitted shirts"
        │
        ▼
┌─ Product Analyst ─────────────────────────────────────┐
│  normalized: "men's cotton knitted shirts"            │
│  category:   apparel                                  │
│  attrs:      type=shirt · material=cotton · knit      │
│  keywords:   [men's shirts, cotton, knitted, apparel] │
└───────────────────────────┬───────────────────────────┘
                            │
                            ▼
┌─ Nomenclature Retriever (1st retrieval) ──────────────┐
│  hybrid search → candidate chapters                   │
│    ch.61  Articles of apparel, knitted or crocheted   │
│    ch.62  Articles of apparel, not knitted…           │
│    …                                                  │
└───────────────────────────┬───────────────────────────┘
                            │
                            ▼
┌─ Research Analyst ────────────────────────────────────┐
│  ranked pathways                                      │
│    1. ch.61  (knitted apparel — primary)              │
│    2. ch.62  (woven apparel — secondary / contrast)   │
└───────────────────────────┬───────────────────────────┘
                            │
                            ▼
┌─ Classification Context Builder (2nd retrieval) ──────┐
│  hybrid search over heading chunks → compact context  │
│    ch.61  notes + 61.05 → 6105.10 (knitted, cotton)   │
│    ch.62  notes + 62.05 → 6205.20 (woven, cotton)     │
└───────────────────────────┬───────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              │  + GIR rules (fixed size) │
              └─────────────┬─────────────┘
                            ▼
┌─ Classification Analyst ──────────────────────────────┐
│  apply chapter notes/headings + GIR                   │
│  1. 6105.10  Men's or boys' shirts, knitted, of cotton│
│     score 1.00 · chapter 61                           │
│  2. 6205.20  Men's or boys' shirts of cotton          │
│     score 0.30 · chapter 62                           │
└───────────────────────────────────────────────────────┘
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
