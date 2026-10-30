"""
Tree-sitter AST & Code Intelligence Graph for cli-agent.
Extracts symbols (classes, functions, methods), signatures, docstrings, and relationships
(calls, inheritance, imports) into a persistent SQLite knowledge graph.
"""

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser

from .memory import DEFAULT_DB_PATH, resolve_project_key

PY_LANGUAGE = Language(tspython.language())


def _node_text(node: Node | None) -> str:
    """Safely decodes node text into a string."""
    if node is None or node.text is None:
        return ""
    return node.text.decode("utf-8", errors="replace")


EXCLUDED_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "build",
    "dist",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "node_modules",
}


@dataclass
class CodeSymbol:
    symbol_name: str
    symbol_type: str  # 'class' | 'function' | 'method'
    start_line: int
    end_line: int
    signature: str = ""
    docstring: str = ""
    file_path: str = ""
    project_key: str = "global"
    id: int | None = None


@dataclass
class SymbolEdge:
    target_symbol_name: str
    relationship_type: str  # 'calls' | 'inherits' | 'imports'
    source_symbol_id: int | None = None
    id: int | None = None


class CodeGraphStore:
    """Persistent SQLite store for AST code symbols and dependency graph edges."""

    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS code_symbols (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_key TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    symbol_name TEXT NOT NULL,
                    symbol_type TEXT NOT NULL,
                    signature TEXT,
                    docstring TEXT,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS symbol_edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_symbol_id INTEGER NOT NULL,
                    target_symbol_name TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    FOREIGN KEY(source_symbol_id) REFERENCES code_symbols(id) ON DELETE CASCADE
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_symbols_lookup ON code_symbols(project_key, symbol_name);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_symbols_file ON code_symbols(project_key, file_path);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_edges_lookup ON symbol_edges(target_symbol_name);"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS file_metadata (
                    project_key TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    mtime REAL NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (project_key, file_path)
                );
                """
            )

    def get_file_mtime(self, project_key: str, file_path: str) -> float | None:
        """Returns the last indexed modification time (st_mtime) for a file, or None if not indexed."""
        norm_path = os.path.normpath(file_path)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT mtime FROM file_metadata WHERE project_key = ? AND file_path = ?",
                (project_key, norm_path),
            )
            row = cursor.fetchone()
            return float(row["mtime"]) if row else None

    def set_file_mtime(self, project_key: str, file_path: str, mtime: float) -> None:
        """Persists the modification time (st_mtime) for a file."""
        norm_path = os.path.normpath(file_path)
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO file_metadata (project_key, file_path, mtime, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(project_key, file_path) DO UPDATE SET
                    mtime = excluded.mtime,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (project_key, norm_path, mtime),
            )
            conn.commit()

    def save_file_symbols(
        self,
        project_key: str,
        file_path: str,
        symbols_with_edges: list[tuple[CodeSymbol, list[SymbolEdge]]],
        mtime: float | None = None,
    ) -> None:
        """Atomically replaces symbols and edges for a given file."""
        norm_path = os.path.normpath(file_path)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Delete old symbols (foreign keys cascade will remove old edges)
            cursor.execute(
                "DELETE FROM code_symbols WHERE project_key = ? AND file_path = ?",
                (project_key, norm_path),
            )

            for sym, edges in symbols_with_edges:
                cursor.execute(
                    """
                    INSERT INTO code_symbols (
                        project_key, file_path, symbol_name, symbol_type,
                        signature, docstring, start_line, end_line
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        project_key,
                        norm_path,
                        sym.symbol_name,
                        sym.symbol_type,
                        sym.signature,
                        sym.docstring,
                        sym.start_line,
                        sym.end_line,
                    ),
                )
                sym_id = cursor.lastrowid
                if sym_id:
                    for edge in edges:
                        cursor.execute(
                            """
                            INSERT INTO symbol_edges (source_symbol_id, target_symbol_name, relationship_type)
                            VALUES (?, ?, ?)
                            """,
                            (sym_id, edge.target_symbol_name, edge.relationship_type),
                        )

            if mtime is not None:
                cursor.execute(
                    """
                    INSERT INTO file_metadata (project_key, file_path, mtime, updated_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(project_key, file_path) DO UPDATE SET
                        mtime = excluded.mtime,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (project_key, norm_path, mtime),
                )
            conn.commit()

    def get_file_symbols(self, project_key: str, file_path: str) -> list[dict[str, Any]]:
        norm_path = os.path.normpath(file_path)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, symbol_name, symbol_type, signature, docstring, start_line, end_line
                FROM code_symbols
                WHERE project_key = ? AND file_path = ?
                ORDER BY start_line ASC
                """,
                (project_key, norm_path),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_outline(self, project_key: str, file_path: str) -> str:
        """Produces a token-efficient structural outline of a file."""
        symbols = self.get_file_symbols(project_key, file_path)
        if not symbols:
            return f"No indexed symbols found for '{file_path}' in project '{project_key}'."

        lines: list[str] = [f"### Structural Outline: {file_path}"]
        for s in symbols:
            indent = "  " if s["symbol_type"] == "method" else ""
            sig = s["signature"] or s["symbol_name"]
            line_span = f"(lines {s['start_line']}-{s['end_line']})"
            lines.append(f"{indent}- [{s['symbol_type'].upper()}] {sig} {line_span}")
            if s["docstring"]:
                first_doc = s["docstring"].split("\n")[0].strip()
                lines.append(f'{indent}    """{first_doc}"""')

        return "\n".join(lines)

    def find_definition(self, project_key: str, symbol_name: str) -> list[dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, file_path, symbol_name, symbol_type, signature, docstring, start_line, end_line
                FROM code_symbols
                WHERE project_key = ? AND (symbol_name = ? OR symbol_name LIKE ?)
                ORDER BY file_path, start_line
                """,
                (project_key, symbol_name, f"%.{symbol_name}"),
            )
            return [dict(row) for row in cursor.fetchall()]

    def find_references(self, project_key: str, symbol_name: str) -> list[dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT e.relationship_type, e.target_symbol_name,
                       s.file_path, s.symbol_name AS caller_name, s.symbol_type AS caller_type,
                       s.start_line, s.end_line
                FROM symbol_edges e
                JOIN code_symbols s ON e.source_symbol_id = s.id
                WHERE s.project_key = ? AND e.target_symbol_name = ?
                ORDER BY s.file_path, s.start_line
                """,
                (project_key, symbol_name),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_symbol_impact(self, project_key: str, symbol_name: str) -> dict[str, Any]:
        """
        Aggregates definition locations, incoming caller references, dependent files,
        and computes a structural modification risk rating ("LOW", "MEDIUM", "HIGH").
        """
        definitions = self.find_definition(project_key, symbol_name)
        references = self.find_references(project_key, symbol_name)

        # Also check references to the short method name if symbol_name is qualified (e.g. Class.method)
        if "." in symbol_name:
            short_name = symbol_name.split(".")[-1]
            short_refs = self.find_references(project_key, short_name)
            existing_keys = {
                (r["file_path"], r["caller_name"], r["relationship_type"], r["start_line"])
                for r in references
            }
            for sr in short_refs:
                key = (
                    sr["file_path"],
                    sr["caller_name"],
                    sr["relationship_type"],
                    sr["start_line"],
                )
                if key not in existing_keys:
                    references.append(sr)
                    existing_keys.add(key)

        dependent_files = sorted({r["file_path"] for r in references})
        ref_count = len(references)
        dep_file_count = len(dependent_files)

        if ref_count >= 5 or dep_file_count >= 3:
            risk_rating = "HIGH"
        elif ref_count >= 2 or dep_file_count >= 2:
            risk_rating = "MEDIUM"
        else:
            risk_rating = "LOW"

        return {
            "symbol_name": symbol_name,
            "project_key": project_key,
            "definitions": definitions,
            "definition_count": len(definitions),
            "references": references,
            "reference_count": ref_count,
            "dependent_files": dependent_files,
            "dependent_file_count": dep_file_count,
            "risk_rating": risk_rating,
        }

    def get_stats(self, project_key: str) -> dict[str, int]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(DISTINCT file_path) FROM code_symbols WHERE project_key = ?",
                (project_key,),
            )
            file_count = int(cursor.fetchone()[0] or 0)

            cursor.execute(
                "SELECT COUNT(*) FROM code_symbols WHERE project_key = ?",
                (project_key,),
            )
            symbol_count = int(cursor.fetchone()[0] or 0)

            cursor.execute(
                """
                SELECT COUNT(*) FROM symbol_edges e
                JOIN code_symbols s ON e.source_symbol_id = s.id
                WHERE s.project_key = ?
                """,
                (project_key,),
            )
            edge_count = int(cursor.fetchone()[0] or 0)

            return {
                "files": file_count,
                "symbols": symbol_count,
                "edges": edge_count,
            }


class CodeGraphIndexer:
    """Parses source files via Tree-sitter and writes structural AST representations to store."""

    def __init__(self, store: CodeGraphStore | None = None):
        self.store = store or CodeGraphStore()
        self.parser = Parser(PY_LANGUAGE)

    def _extract_docstring(self, body_node: Node | None) -> str:
        if not body_node:
            return ""
        for child in body_node.children:
            if child.type == "expression_statement":
                for grand in child.children:
                    if grand.type == "string":
                        raw = _node_text(grand)
                        if (raw.startswith('"""') and raw.endswith('"""')) or (
                            raw.startswith("'''") and raw.endswith("'''")
                        ):
                            return raw[3:-3].strip()
                        if (raw.startswith('"') and raw.endswith('"')) or (
                            raw.startswith("'") and raw.endswith("'")
                        ):
                            return raw[1:-1].strip()
                        return raw.strip()
            elif child.type not in ("comment", "\n"):
                break
        return ""

    def _extract_calls(self, node: Node) -> list[str]:
        calls: list[str] = []

        def traverse(curr: Node) -> None:
            if curr.type == "call":
                fn = curr.child_by_field_name("function")
                if fn:
                    if fn.type == "identifier":
                        calls.append(_node_text(fn))
                    elif fn.type == "attribute":
                        attr = fn.child_by_field_name("attribute")
                        if attr:
                            calls.append(_node_text(attr))
            for ch in curr.children:
                traverse(ch)

        traverse(node)
        return calls

    def parse_python_source(
        self, code: str, file_path: str, project_key: str
    ) -> list[tuple[CodeSymbol, list[SymbolEdge]]]:
        """Parses Python source code into symbols and dependency edges."""
        tree = self.parser.parse(code.encode("utf-8"))
        results: list[tuple[CodeSymbol, list[SymbolEdge]]] = []

        for child in tree.root_node.children:
            if child.type == "class_definition":
                name_node = child.child_by_field_name("name")
                if not name_node:
                    continue
                class_name = _node_text(name_node)
                body_node = child.child_by_field_name("body")
                doc = self._extract_docstring(body_node)
                start_l = child.start_point[0] + 1
                end_l = child.end_point[0] + 1

                # Superclasses (inherits edges)
                edges: list[SymbolEdge] = []
                supers_node = child.child_by_field_name("superclasses")
                super_text = ""
                if supers_node:
                    super_text = _node_text(supers_node)
                    for base in supers_node.children:
                        if base.type == "identifier":
                            edges.append(
                                SymbolEdge(
                                    target_symbol_name=_node_text(base),
                                    relationship_type="inherits",
                                )
                            )

                sig = f"class {class_name}{super_text}"
                class_sym = CodeSymbol(
                    symbol_name=class_name,
                    symbol_type="class",
                    start_line=start_l,
                    end_line=end_l,
                    signature=sig,
                    docstring=doc,
                    file_path=file_path,
                    project_key=project_key,
                )
                results.append((class_sym, edges))

                # Process class methods
                if body_node:
                    for bchild in body_node.children:
                        if bchild.type == "function_definition":
                            mname_node = bchild.child_by_field_name("name")
                            if not mname_node:
                                continue
                            mname = _node_text(mname_node)
                            mbody = bchild.child_by_field_name("body")
                            mdoc = self._extract_docstring(mbody)
                            mparams = bchild.child_by_field_name("parameters")
                            mret = bchild.child_by_field_name("return_type")
                            p_str = _node_text(mparams) if mparams else "()"
                            r_str = f" -> {_node_text(mret)}" if mret else ""
                            msig = f"def {mname}{p_str}{r_str}"

                            mcalls = self._extract_calls(bchild)
                            medges = [
                                SymbolEdge(
                                    target_symbol_name=call_target, relationship_type="calls"
                                )
                                for call_target in mcalls
                            ]

                            msym = CodeSymbol(
                                symbol_name=f"{class_name}.{mname}",
                                symbol_type="method",
                                start_line=bchild.start_point[0] + 1,
                                end_line=bchild.end_point[0] + 1,
                                signature=msig,
                                docstring=mdoc,
                                file_path=file_path,
                                project_key=project_key,
                            )
                            results.append((msym, medges))

            elif child.type == "function_definition":
                name_node = child.child_by_field_name("name")
                if not name_node:
                    continue
                fname = _node_text(name_node)
                fbody = child.child_by_field_name("body")
                fdoc = self._extract_docstring(fbody)
                fparams = child.child_by_field_name("parameters")
                fret = child.child_by_field_name("return_type")
                p_str = _node_text(fparams) if fparams else "()"
                r_str = f" -> {_node_text(fret)}" if fret else ""
                fsig = f"def {fname}{p_str}{r_str}"

                fcalls = self._extract_calls(child)
                fedges = [
                    SymbolEdge(target_symbol_name=call_target, relationship_type="calls")
                    for call_target in fcalls
                ]

                fsym = CodeSymbol(
                    symbol_name=fname,
                    symbol_type="function",
                    start_line=child.start_point[0] + 1,
                    end_line=child.end_point[0] + 1,
                    signature=fsig,
                    docstring=fdoc,
                    file_path=file_path,
                    project_key=project_key,
                )
                results.append((fsym, fedges))

            elif child.type in ("import_statement", "import_from_statement"):
                # Track top-level imports as file-level module symbols
                imported_text = _node_text(child).strip()
                import_sym = CodeSymbol(
                    symbol_name="<imports>",
                    symbol_type="import",
                    start_line=child.start_point[0] + 1,
                    end_line=child.end_point[0] + 1,
                    signature=imported_text,
                    file_path=file_path,
                    project_key=project_key,
                )
                import_edges: list[SymbolEdge] = []
                for sub in child.children:
                    if sub.type == "dotted_name":
                        import_edges.append(
                            SymbolEdge(
                                target_symbol_name=_node_text(sub),
                                relationship_type="imports",
                            )
                        )
                results.append((import_sym, import_edges))

        return results

    def index_file(self, file_path: Path | str, project_key: str | None = None) -> int:
        """Indexes a single file into the code graph store."""
        path = Path(file_path).resolve()
        if not path.is_file() or path.suffix != ".py":
            return 0

        p_key = project_key or resolve_project_key()
        try:
            st_mtime = path.stat().st_mtime
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return 0

        rel_path = str(path)
        symbols_with_edges = self.parse_python_source(content, rel_path, p_key)
        self.store.save_file_symbols(p_key, rel_path, symbols_with_edges, mtime=st_mtime)
        return len(symbols_with_edges)

    def index_directory(
        self,
        dir_path: Path | str,
        project_key: str | None = None,
        max_files: int = 500,
    ) -> int:
        """Recursively indexes all Python files under dir_path, skipping unchanged files via st_mtime."""
        root = Path(dir_path).resolve()
        if not root.is_dir():
            return 0

        p_key = project_key or resolve_project_key()
        indexed_count = 0

        for current_root, dirs, files in os.walk(root):
            # Prune excluded directories in-place
            dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS and not d.startswith(".")]

            for file in files:
                if file.endswith(".py"):
                    full_path = (Path(current_root) / file).resolve()
                    try:
                        current_mtime = full_path.stat().st_mtime
                    except OSError:
                        continue

                    cached_mtime = self.store.get_file_mtime(p_key, str(full_path))
                    if cached_mtime is not None and cached_mtime == current_mtime:
                        continue

                    self.index_file(full_path, p_key)
                    indexed_count += 1
                    if indexed_count >= max_files:
                        return indexed_count

        return indexed_count
