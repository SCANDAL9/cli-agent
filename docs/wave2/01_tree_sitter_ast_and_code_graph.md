# Specification: Tree-sitter AST & Code Intelligence Graph

## 1. Problem Statement & Baseline Limitations

In Wave 1, `search_code` relied on regex/text line scans:
- **Zero Semantic Context:** Cannot differentiate between a function definition, an invocation, a string literal, or a comment.
- **Blind Refactoring:** Modifying a function signature cannot trace all callers across the codebase.
- **Context Explosion:** LLMs must read entire 1000-line files simply to discover function signatures and export interfaces.

---

## 2. Proposed Architecture: AST & Code Graph

```
                               Source Code Files (.py, .ts, .go, .rs)
                                                 │
                                                 ▼
                               ┌───────────────────────────────────┐
                               │       Tree-sitter Language        │
                               │          Grammar Parsers          │
                               └─────────────────┬─────────────────┘
                                                 │ Concrete Syntax Trees (CST)
                                                 ▼
                               ┌───────────────────────────────────┐
                               │      Symbol Extraction Engine     │
                               │ - Functions, Classes, Methods     │
                               │ - Imports, Exports, Signatures    │
                               └─────────────────┬─────────────────┘
                                                 │
                                                 ▼
                               ┌───────────────────────────────────┐
                               │   Code Dependency Graph (SQLite)  │
                               │ - Nodes: Symbols & Files          │
                               │ - Edges: Calls, Imports, Inherits │
                               └─────────────────┬─────────────────┘
                                                 │
                         ┌───────────────────────┴───────────────────────┐
                         ▼                                               ▼
          ┌─────────────────────────────┐                 ┌─────────────────────────────┐
          │   Model Tool: `find_defs`   │                 │ `get_symbol_outline`        │
          │ Locate definition & types   │                 │ File skeleton (0 token cost)│
          └─────────────────────────────┘                 └─────────────────────────────┘
```

---

## 3. Data Model: Code Knowledge Graph (SQLite)

```sql
CREATE TABLE IF NOT EXISTS code_symbols (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_key TEXT NOT NULL,
    file_path TEXT NOT NULL,
    symbol_name TEXT NOT NULL,
    symbol_type TEXT NOT NULL,     -- 'class' | 'function' | 'method' | 'interface'
    signature TEXT,                -- 'def calculate(total: float) -> float'
    docstring TEXT,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS symbol_edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_symbol_id INTEGER NOT NULL,
    target_symbol_name TEXT NOT NULL,
    relationship_type TEXT NOT NULL, -- 'calls' | 'imports' | 'inherits' | 'instantiates'
    FOREIGN KEY(source_symbol_id) REFERENCES code_symbols(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_symbols_lookup ON code_symbols(project_key, symbol_name);
CREATE INDEX IF NOT EXISTS idx_symbols_file ON code_symbols(project_key, file_path);
```

---

## 4. New Agentic Tools Specifications

### 4.1. `get_symbol_outline(file_path: str)`
Returns a lightweight skeleton of a source file showing only classes, functions, docstrings, and signatures without line-by-line implementation bodies.
- **Benefit:** Consumes ~120 tokens instead of 2500 tokens for full file reads.

### 4.2. `find_symbol_definition(symbol_name: str)`
Instantly jumps to the exact file, start line, and declaration of any symbol in the repository.

### 4.3. `find_symbol_references(symbol_name: str)`
Returns all call sites, instantiations, and imports of a symbol across the entire project graph.

---

## 5. Implementation Roadmap

1. Add `tree-sitter` and `tree-sitter-language-pack` to dependencies.
2. Build `CodeGraphIndexer` that crawls project source trees and indexes symbols on startup or file edit.
3. Register AST tools in `TOOL_REGISTRY` and `GEMINI_TOOLS_DECLARATIONS`.
4. Add unit and integration tests verifying graph traversal and cross-references.
