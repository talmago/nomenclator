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

1. 8708.99 — Other parts and accessories of motor vehicles
   Score: 0.95
   Source chapter: 87
   Reasoning:
     • The product is a lithium ion battery pack designed for electric vehicles.
     • Chapter 87 covers vehicles and their parts.
     • Subheading 8708.99 covers other motor vehicle parts and accessories.
     • The battery pack is designed specifically for electric vehicles.

2. 8507.60 — Lithium-ion accumulators
   Score: 0.75
   Source chapter: 85
   Reasoning:
     • Lithium-ion batteries are generally classified under heading 8507.
     • Because this battery pack is designed specifically for electric vehicles,
       Chapter 87 is considered the stronger classification.
     • Heading 8507 remains a plausible alternative depending on the product's
       presentation and intended use.

Token usage
  Prompt tokens:     9,862
  Completion tokens: 997
  Total tokens:      10,859
  Estimated cost:    $0.00554
```

The pipeline performs the following steps:

1. Extract structured product facts.
2. Retrieve relevant HS chapters using hybrid semantic and lexical search.
3. Rank the retrieved chapters.
4. Analyze chapter notes and headings.
5. Produce ranked HS code candidates.

## Development

Common development commands are available through the project's `Makefile`.

```bash
make install      # Install dependencies
make format       # Format the source code
make lint         # Run static analysis
make test         # Execute the test suite
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
