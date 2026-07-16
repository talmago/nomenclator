# AGENTS.md

Compact map of `src/nomenclator/` for coding agents. 

For pipeline stages, design principles, and component responsibilities, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Layout

```
src/nomenclator/
├── agent.py          # Orchestrates the HS classification pipeline
├── models/           # Shared Pydantic/dataclass contracts between stages
├── nomenclature/     # Download, parse, and cache official HS data → HSTree
├── retrieval/        # Domain-agnostic hybrid search (embeddings + BM25 + RRF)
├── signatures/       # DSPy signatures for each reasoning stage
└── usage.py          # Token usage / cost helpers from DSPy LM history
```

## Notes

- `agent.py` wires stages; business logic lives in nomenclature, retrieval, and signatures.
- `nomenclature/` is independent of DSPy and retrieval.
- `retrieval/` is domain-agnostic; HS adaptation is converting `HSTree` into documents.
- `models/` are the interfaces between pipeline stages.
