# AGENTS.md

Guidelines for coding agents working on **Nomenclator**.

For architecture and pipeline details, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Coding Principles

### Think Before Coding

- Understand the existing design before changing it.
- Surface meaningful tradeoffs or inconsistencies.
- Ask for clarification only when ambiguity materially affects the implementation.
- Prefer extending existing components over introducing new ones.

### Keep It Simple

- Implement the smallest change that solves the problem.
- Avoid speculative abstractions and unnecessary configurability.
- Follow existing patterns before introducing new ones.
- Optimize for readability rather than cleverness.

### Make Surgical Changes

- Modify only code relevant to the requested task.
- Avoid unrelated refactors or formatting-only changes.
- Remove dead code introduced by your changes.
- Keep public APIs stable unless explicitly requested.

### Validate Your Work

Before considering a task complete:

- run the relevant tests;
- run formatting, linting, and type checking when applicable;
- ensure new behavior is covered by tests whenever practical.

Never claim something was tested unless it actually was.

## Development

Typical commands:

```bash
make format
make lint
make typecheck
make test
```

## Project Structure

```
src/nomenclator/
├── agent.py          # HS classification pipeline orchestration
├── models/           # Shared contracts between pipeline stages
├── nomenclature/     # WCO HS retrieval, parsing and caching
├── retrieval/        # Domain-agnostic hybrid retrieval engine
├── signatures/       # DSPy reasoning stages
└── usage.py          # DSPy token usage helpers
```

### Responsibilities

- `agent.py` orchestrates the pipeline only.
- `retrieval/` is generic and independent of HS classification.
- `nomenclature/` owns WCO parsing and document loading.
- `models/` define contracts between stages.
- `signatures/` contain DSPy signatures and prompting.
- Business logic belongs in implementation modules, not orchestration.

## Design Principles

The architecture intentionally separates deterministic processing from LLM reasoning.

- Retrieval should remain deterministic.
- DSPy modules should reason over retrieved context, not perform retrieval.
- Components should remain independently testable.
- Prefer explicit data flow over hidden coupling.
- Favor composition over large monolithic classes.
