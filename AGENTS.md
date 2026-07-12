# AGENTS.md

This document describes the major components of the project and their
responsibilities.

## Project Structure

```
src/nomenclator/
├── agent.py
├── models/
├── nomenclature/
├── retrieval/
├── signatures/
└── usage.py
```

## Components

### `agent.py`

Implements the end-to-end HS classification pipeline.

The pipeline consists of four stages:

1. Product Analyst
2. Hybrid retrieval
3. Research Analyst
4. Classification Analyst

The agent orchestrates these stages but contains very little business logic.
Retrieval and reasoning are delegated to dedicated components.

---

### `models/`

Shared data models used throughout the project.

Models are primarily implemented as Pydantic models and dataclasses describing:

- parsed HS nomenclature
- product facts
- retrieval results
- navigation results
- classification results
- token usage

These models form the interface between pipeline stages.

---

### `nomenclature/`

Downloads, parses, and caches the official HS nomenclature.

Responsibilities include:

- downloading source PDFs
- parsing sections, chapters, headings, and notes
- constructing the `HSTree`
- exposing a simple client interface

This package is completely independent from DSPy and retrieval.

---

### `retrieval/`

Generic hybrid retrieval components.

The retrieval layer is intentionally domain-agnostic.

It provides:

- semantic retrieval using sentence embeddings
- BM25 lexical retrieval
- hybrid ranking using Reciprocal Rank Fusion (RRF)

The HS-specific adaptation consists only of converting the `HSTree`
into retrieval documents.

---

### `signatures/`

DSPy signatures defining each reasoning stage.

Each signature specifies:

- inputs
- outputs
- instructions
- reasoning constraints

Current signatures include:

- `ProductAnalystSignature`
- `ResearchAnalystSignature`
- `ClassificationAnalystSignature`

---

### `usage.py`

Utilities for collecting token usage and cost information from DSPy language
model history.

This module is independent from the classification pipeline and is intended
for benchmarking and monitoring.

## Design Principles

The project follows a layered architecture.

```
Nomenclature parsing
        │
        ▼
Navigation retrieval
        │
        ▼
LLM reasoning
        │
        ▼
HS classification
```

Each layer has a single responsibility.

- **Nomenclature** parses official HS data.
- **Retrieval** performs deterministic document search.
- **DSPy agents** perform reasoning.
- **Models** define the contracts between layers.

This separation makes the retrieval layer independently testable and allows
reasoning quality to be evaluated separately from retrieval quality.