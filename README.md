# Nomenclator

**Nomenclator** is a Python library for Harmonized System (HS) classification that combines deterministic retrieval over the official HS nomenclature with structured LLM reasoning.

The library uses a multi-stage pipeline to progressively narrow the search space—from product analysis, through chapter and heading retrieval, to legally grounded HS classification.

## Features

- 🌳 Parses the official HS 2022 nomenclature into a structured tree
- 🔎 Two-stage hybrid retrieval over chapters and headings (semantic + BM25)
- 🤖 Multi-stage DSPy pipeline for product analysis, research, and classification
- ⚖️ Applies chapter notes and the General Rules for Interpretation (GIR) during classification
- 📊 Built-in benchmarking framework for evaluating classification quality
- 🧩 Modular architecture with typed intermediate models and independently testable components

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

Install the package and classify a product directly from the command line:

```bash
nomenclator "Men's cotton knitted shirts"
```

Example output:

```text
╭────────────────────────────────────────────────────────────────────────╮
│ Nomenclator (openai/gpt-4.1-mini)                                      │
╰────────────────────────────────────────────────────────────────────────╯

Product
  Men's cotton knitted shirts

╭───────────── Best Classification────────────────────────────╮
│ HS Code       6105.10                                       │
│ Confidence    1.00                                          │
│ Description   Men's or boys' shirts, knitted or crocheted — Of cotton  │
│ Chapter       6101-2022E                                    │
╰─────────────────────────────────────────────────────────────╯

Reasoning
├── Product is a knitted cotton shirt for men, matching heading 61.05 and subheading 6105.10 criteria.
├── Chapter 61 notes specify applicability to knitted or crocheted garments, confirming the correct chapter.
├── Heading 61.05 is the most specific heading for men's knitted shirts, consistent with GIR 3(a).
├── Other headings, including those for woven shirts or women's garments, do not apply.
└── No competing heading provides a more specific classification.

╭─────────────────────── Performance ────────────────────────╮
│ Prompt tokens          18,053                              │
│ Completion tokens      648                                 │
│ Total tokens           18,701                              │
│ Estimated cost        $0.00826                             │
╰────────────────────────────────────────────────────────────╯
```

The CLI also accepts input from standard input:

```bash
echo "Fresh bananas" | nomenclator
```

Run `nomenclator --help` to see all available options.

---

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

Completed:

- ✅ Two-stage hybrid retrieval (chapter and heading levels)
- ✅ Multiple embedding backends (Sentence Transformers and FastEmbed)
- ✅ Classification benchmarking and evaluation framework

Planned improvements:

- DSPy optimization using evaluation datasets
- Interactive terminal UI (TUI) for product classification and result exploration
- Optional audit stage for reviewing classification results
- Support for additional customs nomenclatures (e.g. TARIC, HTSUS)

## License

MIT License.
