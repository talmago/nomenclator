# Architecture

## Overview

Nomenclator is a hybrid retrieval and reasoning system for Harmonized System (HS)
classification.

The pipeline separates deterministic retrieval from LLM-based analysis. Product
descriptions are first transformed into structured product facts and
HS-oriented retrieval terms, which drive a two-stage retrieval process over the
HS nomenclature. DSPy-powered analysts then perform structured reasoning over
the retrieved context.

This separation allows retrieval quality and classification quality to be
evaluated and improved independently.

## Pipeline

```
Raw product description
        │
        ▼
┌────────────────────────┐
│   Product Analyst      │  normalize product identity
│                        │  extract attributes
│                        │  generate HS retrieval terms
└────────────┬───────────┘
             │
             ▼
   product facts + retrieval query
             │
             ▼
┌────────────────────────┐
│   Chapter Retriever    │  hybrid search (semantic + BM25)
└────────────┬───────────┘
             │
             ▼
      candidate chapters
             │
             ▼
┌────────────────────────┐
│  Research Analyst      │  evaluate classification pathways
└────────────┬───────────┘
             │
             ▼
    shortlisted chapters
             │
             ▼
┌────────────────────────┐
│   Heading Retriever    │  second-stage hybrid retrieval
└────────────┬───────────┘
             │
             ▼
 compact chapter context
 (notes + selected headings)
             │
             ▼
┌────────────────────────┐
│ Classification Analyst │  apply GIR and classify
└────────────┬───────────┘
             │
             ▼
     Ranked HS classifications
```

## Components

### Product Analyst

The Product Analyst transforms an unstructured product description into
structured product facts suitable for retrieval and classification.

Responsibilities:

- Normalize the product's intrinsic identity.
- Extract classification-relevant attributes.
- Infer broader technical or commercial product categories.
- Generate HS-oriented retrieval terms that bridge commercial language to
  Harmonized System terminology.
- Preserve user-provided HS code hints.

Output:

- Structured product facts.
- Retrieval query.

The Product Analyst does **not** perform classification.

---

### Chapter Retriever

The Chapter Retriever performs deterministic candidate generation over the HS
nomenclature.

It combines:

- Semantic retrieval using sentence embeddings.
- BM25 lexical retrieval.
- Reciprocal Rank Fusion (RRF).

Output:

- Ranked candidate chapters.

The retrieval layer is deterministic and independent from LLM reasoning.

---

### Research Analyst

The Research Analyst evaluates retrieved chapters and identifies the most
plausible classification pathways.

Responsibilities:

- Review candidate chapters.
- Compare competing classification areas.
- Rank the most relevant chapters.
- Explain why each chapter is relevant.

Output:

- Shortlisted chapters.

The Research Analyst does **not** assign HS codes.

---

### Heading Retriever

The Heading Retriever performs a second retrieval stage within the shortlisted
chapters to construct a compact classification context.

Responsibilities:

- Split shortlisted chapters into heading-level chunks.
- Retrieve the globally most relevant heading chunks.
- Always include complete chapter notes.
- Respect a configurable global chunk budget.

Output:

- Compact chapter context containing chapter metadata, notes, and selected
  heading hierarchies.

Like the Chapter Retriever, it uses semantic retrieval, BM25, and Reciprocal
Rank Fusion.

General Rules for the Interpretation of the Harmonized System (GIR) are loaded
separately as fixed-size context.

---

### Classification Analyst

The Classification Analyst performs the final legal reasoning.

Responsibilities:

- Analyze structured product facts.
- Review chapter notes and heading context.
- Apply the General Rules for Interpretation (GIR).
- Evaluate competing classifications.
- Produce ranked HS code candidates with supporting reasoning.

Output:

- Ranked HS classifications.
- Confidence scores.
- Classification reasoning.

---

## Design Principles

### Separation of Retrieval and Reasoning

Retrieval identifies where to look.

LLM analysts determine why a classification is correct.

Each layer can evolve independently.

### Retrieval-Oriented Product Analysis

The Product Analyst translates commercial product descriptions into terminology
that aligns with the structure of the Harmonized System. This semantic bridge
improves retrieval quality without requiring the retrieval index to encode
large synonym vocabularies.

### Deterministic Retrieval

The retrieval layer produces consistent results for the same nomenclature,
embedding model, and query.

### Structured Agent Outputs

All analyst stages return typed outputs that can be reliably consumed by the
next stage.

### Progressive Narrowing

The pipeline progressively narrows the search space:

1. Raw product description.
2. Structured product facts and retrieval terms.
3. Candidate HS chapters.
4. Candidate heading context.
5. Ranked HS classifications.

Retrieval is performed twice—first over chapters and then over heading
chunks—allowing the reasoning model to operate over a compact, highly relevant
subset of the HS nomenclature.