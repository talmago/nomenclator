# Architecture

## Overview

Nomenclator is designed as a hybrid retrieval and reasoning system for
Harmonized System (HS) classification.

The architecture separates deterministic information retrieval from LLM-based
analysis. The retrieval layer identifies relevant areas of the HS nomenclature,
while DSPy-powered analysts perform structured reasoning over the retrieved
context.

This separation allows retrieval quality and classification reasoning quality
to be evaluated independently.

## Pipeline

```
Raw product description
        │
        ▼
┌───────────────────────┐
│   Product Analyst     │  extract structured facts
└───────────┬───────────┘
            │
            ▼
   product facts (description · attributes · keywords)
            │
            ▼
┌───────────────────────┐
│ Nomenclature Retriever│  semantic + BM25 (RRF)
└───────────┬───────────┘
            │
            ▼
   candidate HS chapters
            │
            ▼
┌───────────────────────┐
│   Research Analyst    │  evaluate classification pathways
└───────────┬───────────┘
            │
            ▼
   ranked chapter candidates
            │
            ▼
┌───────────────────────┐
│ Classification Context│  global heading-chunk retrieval
│        Builder        │
└───────────┬───────────┘
            │
            ▼
   compact chapter context (notes + selected headings)
            │
            ▼
┌───────────────────────┐
│ Classification Analyst│  apply HS rules and reasoning
└───────────┬───────────┘
            │
            ▼
   HS code candidates
```

## Components

### Product Analyst

The Product Analyst transforms an unstructured product description into
structured product facts.

Its responsibilities:

- Normalize the product description.
- Extract classification-relevant attributes.
- Identify useful retrieval keywords.
- Preserve user-provided HS code hints as unverified signals.

The Product Analyst does **not** perform classification.

Output:

- Normalized description.
- Product attributes.
- Retrieval keywords.

---

### Nomenclature Retriever

The Nomenclature Retriever provides deterministic candidate generation over the
HS nomenclature.

It combines two complementary retrieval strategies:

- **Semantic retrieval** using sentence embeddings to identify conceptually
  similar HS chapters.
- **BM25 lexical retrieval** to preserve exact terminology matches.

The two rankings are combined using **Reciprocal Rank Fusion (RRF)**, providing
robust retrieval across both conceptual and terminology-driven queries.

Output:

- Ranked candidate HS chapters.

The retrieval layer is independent from LLM reasoning, allowing retrieval
performance to be measured separately from classification performance.

---

### Research Analyst

The Research Analyst evaluates retrieved HS chapters and identifies plausible
classification pathways.

Its responsibilities:

- Review retrieved chapters.
- Compare competing classification areas.
- Rank the most relevant chapters.
- Explain why each chapter is relevant.

The Research Analyst does **not** assign final HS codes.

Output:

- Ranked chapter candidates.

---

### Classification Context Builder

The Classification Context Builder compacts the shortlisted chapters into a
prompt-sized context for the Classification Analyst. It sits between the
Research Analyst and the Classification Analyst and is the second retrieval
stage of the pipeline.

Its responsibilities:

- Split each shortlisted chapter into heading-level chunks (one heading plus
  its subheadings per chunk).
- Index the chunks of **all** shortlisted chapters together in a single hybrid
  retriever.
- Retrieve the globally most relevant chunks, bounded by a global chunk
  budget (`max_chunks`).
- Guarantee at least one heading chunk per shortlisted chapter (the floor) so
  the Research Analyst's shortlisting is respected.
- Always include the full chapter notes for every shortlisted chapter,
  regardless of which chunks were retrieved, because notes apply to the whole
  chapter and are required for legal reasoning.

The global budget bounds the total Classification Analyst prompt size
regardless of how many chapters were shortlisted or how dense they are. This
trades classification accuracy against model cost: a smaller budget yields a
more compact prompt and lower token cost but risks dropping the heading that
contains the correct 6-digit subheading, while a larger budget improves recall
at the expense of a larger prompt.

Output:

- One compact context entry per shortlisted chapter, containing chapter
  metadata, all chapter notes, and the selected heading hierarchy.

The context builder is deterministic and reuses the same hybrid retrieval
machinery (semantic + BM25 with RRF) as the Nomenclature Retriever, applied
this time to heading chunks instead of whole chapters.

General Rules for the Interpretation of the Harmonized System (GIR) are loaded
separately as fixed-size context for the Classification Analyst; they are not
part of the heading-budgeted chapter context.

---

### Classification Analyst

The Classification Analyst performs the final classification reasoning.

Its responsibilities:

- Analyze product facts.
- Review the compact classification context (chapter notes and selected
  headings).
- Apply the provided General Rules for Interpretation (GIR) when resolving
  conflicts, incomplete goods, mixtures, and related legal questions.
- Consider competing classifications.
- Produce ranked HS code candidates with reasoning.

Output:

- Candidate HS codes.
- Classification reasoning.
- Confidence scores.

---

## Design Principles

### Separation of Retrieval and Reasoning

Retrieval and reasoning are intentionally decoupled:

- Retrieval identifies where to look.
- Analysts determine why a classification is appropriate.

This allows each layer to evolve independently.

### Deterministic Retrieval

The retrieval layer does not depend on LLM output.

Given the same nomenclature data, embedding model, and query, retrieval produces
consistent candidate chapters.

### Structured Agent Outputs

All LLM stages use typed outputs to ensure that intermediate results remain
machine-readable and can be passed reliably between pipeline stages.

### Progressive Narrowing

The system follows a narrowing strategy:

1. Raw product description.
2. Structured product facts.
3. Candidate HS chapters.
4. Globally selected heading chunks (bounded by a global chunk budget).
5. HS code candidates.

Retrieval is applied twice, at decreasing granularity: first over whole
chapters to shortlist classification areas, then over heading chunks within
the shortlisted chapters to select only the relevant headings. This reduces
the amount of nomenclature context required by the reasoning model while
keeping the classification process grounded in HS data and bounding the total
prompt size.
