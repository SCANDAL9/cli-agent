# Specification: Hybrid Code Search (FTS5 + Dense Embeddings + RRF)

## 1. Context & Motivation

In large codebases:
- **BM25 / FTS5 Limitations:** Excellent for exact identifier names (`TokenProvider`, `handle_auth`), but fails on semantic/intent queries (*"where is the rate limit handled when calling external APIs?"*).
- **Dense Vector Limitations:** Excellent at conceptual similarity, but frequently misses exact symbol names, line numbers, and variable names.

Wave 2 introduces **Hybrid Code Search** merging sparse BM25 with dense semantic vector embeddings.

---

## 2. Architecture & Reciprocal Rank Fusion (RRF)

```
                            User Code Search Query
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
        Sparse Retrieval (BM25)               Dense Retrieval (Vertex Vector)
         - SQLite FTS5 index                   - text-embedding-005
         - Exact identifiers                   - Semantic intent
                    │                                     │
                    └──────────────────┬──────────────────┘
                                       │
                                       ▼
                       Reciprocal Rank Fusion (RRF)
                 Score = 1 / (60 + r_bm25) + 1 / (60 + r_vector)
                                       │
                                       ▼
                       Top-K AST Code Chunk Results
```

---

## 3. AST-Aware Code Chunking Strategy

Traditional text chunking splits text every 500 characters, splitting functions in half.
Wave 2 uses **AST-Aware Chunking**:
- Chunks correspond to logical code units (a full class definition, an entire method, or a module header).
- Every chunk includes metadata: file path, enclosing class name, parameter types, and docstrings.
