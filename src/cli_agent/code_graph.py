"""
Tree-sitter AST & Code Intelligence Graph for cli-agent.
Extracts symbols (classes, functions, methods), signatures, docstrings, and relationships
(calls, inheritance, imports) into a persistent SQLite knowledge graph.
"""

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser, Query, QueryCursor

from .config import StorageConfig
from .memory import DEFAULT_DB_PATH, resolve_project_key
from .storage import StorageConnection, StorageEngine, resolve_store_engine

PY_LANGUAGE = Language(tspython.language())
PY_CALL_QUERY = Query(
    PY_LANGUAGE,
    """
    (call function: (identifier) @function) @call
    (call function: (attribute attribute: (identifier) @function)) @call
    """,
)

SUPPORTED_EXTENSIONS = {".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs"}


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

    def __init__(
        self,
        db_path: Path | str | None = None,
        *,
        storage_engine: StorageEngine | None = None,
        storage_config: StorageConfig | None = None,
    ):
        self.storage_engine = resolve_store_engine(
            DEFAULT_DB_PATH,
            Path(db_path) if db_path is not None else None,
            storage_engine,
            storage_config,
        )
        self.db_path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> StorageConnection:
        return self.storage_engine.connect()

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
                "CREATE INDEX IF NOT EXISTS idx_edges_source ON symbol_edges(source_symbol_id);"
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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS file_graph_state (
                    project_key TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    graph_digest TEXT NOT NULL,
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

    def remove_file_symbols(self, project_key: str, file_path: str) -> None:
        """Removes one file's graph records and incremental-index metadata atomically."""
        norm_path = os.path.normpath(file_path)
        with self._get_connection() as conn:
            conn.execute(
                "DELETE FROM code_symbols WHERE project_key = ? AND file_path = ?",
                (project_key, norm_path),
            )
            conn.execute(
                "DELETE FROM file_metadata WHERE project_key = ? AND file_path = ?",
                (project_key, norm_path),
            )
            conn.execute(
                "DELETE FROM file_graph_state WHERE project_key = ? AND file_path = ?",
                (project_key, norm_path),
            )
            conn.commit()

    def save_file_symbols(
        self,
        project_key: str,
        file_path: str,
        symbols_with_edges: list[tuple[CodeSymbol, list[SymbolEdge]]],
        mtime: float | None = None,
    ) -> None:
        """Atomically synchronizes one file's symbols and edges."""
        self.save_files_symbols([(project_key, file_path, symbols_with_edges, mtime)])

    def save_files_symbols(
        self,
        files: list[tuple[str, str, list[tuple[CodeSymbol, list[SymbolEdge]]], float | None]],
    ) -> None:
        """Atomically synchronizes several files in a single storage transaction."""
        if not files:
            return
        with self._get_connection() as conn:
            conn.execute(self.storage_engine.migration_transaction_sql)
            metadata_updates: list[tuple[str, str, float]] = []
            graph_state_updates: list[tuple[str, str, str]] = []
            for project_key, file_path, symbols_with_edges, mtime in files:
                self._save_file_symbols_in_transaction(
                    conn,
                    project_key,
                    file_path,
                    symbols_with_edges,
                    mtime,
                    metadata_updates,
                    graph_state_updates,
                )
            if metadata_updates:
                conn.executemany(
                    """
                    INSERT INTO file_metadata (project_key, file_path, mtime, updated_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(project_key, file_path) DO UPDATE SET
                        mtime = excluded.mtime,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    metadata_updates,
                )
            if graph_state_updates:
                conn.executemany(
                    """
                    INSERT INTO file_graph_state (project_key, file_path, graph_digest)
                    VALUES (?, ?, ?)
                    ON CONFLICT(project_key, file_path) DO UPDATE SET
                        graph_digest = excluded.graph_digest
                    """,
                    graph_state_updates,
                )

    def _save_file_symbols_in_transaction(
        self,
        conn: StorageConnection,
        project_key: str,
        file_path: str,
        symbols_with_edges: list[tuple[CodeSymbol, list[SymbolEdge]]],
        mtime: float | None,
        metadata_updates: list[tuple[str, str, float]],
        graph_state_updates: list[tuple[str, str, str]],
    ) -> None:
        norm_path = os.path.normpath(file_path)
        next_graph = [
            (
                (
                    sym.symbol_name,
                    sym.symbol_type,
                    sym.signature,
                    sym.docstring,
                    sym.start_line,
                    sym.end_line,
                ),
                [(edge.target_symbol_name, edge.relationship_type) for edge in edges],
            )
            for sym, edges in symbols_with_edges
        ]
        graph_digest = hashlib.blake2b(
            json.dumps(next_graph, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
            digest_size=16,
        ).hexdigest()
        state = conn.execute(
            "SELECT graph_digest FROM file_graph_state WHERE project_key = ? AND file_path = ?",
            (project_key, norm_path),
        ).fetchone()
        if state is not None and state["graph_digest"] == graph_digest:
            if mtime is not None:
                metadata_updates.append((project_key, norm_path, mtime))
            return

        cursor = conn.execute(
            """
            SELECT s.id, s.symbol_name, s.symbol_type, s.signature, s.docstring,
                   s.start_line, s.end_line, e.id AS edge_id,
                   e.target_symbol_name, e.relationship_type
            FROM code_symbols s
            LEFT JOIN symbol_edges e ON e.source_symbol_id = s.id
            WHERE s.project_key = ? AND s.file_path = ?
            ORDER BY s.start_line, s.end_line, s.id, e.id
            """,
            (project_key, norm_path),
        )
        existing_records: list[dict[str, Any]] = []
        previous_id = None
        current_record: dict[str, Any] | None = None
        for row in cursor.fetchall():
            if row["id"] != previous_id:
                current_record = {
                    "id": row["id"],
                    "fields": (
                        row["symbol_name"],
                        row["symbol_type"],
                        row["signature"],
                        row["docstring"],
                        row["start_line"],
                        row["end_line"],
                    ),
                    "edges": [],
                    "edge_records": [],
                }
                existing_records.append(current_record)
                previous_id = row["id"]
            if row["target_symbol_name"] is not None:
                assert current_record is not None
                current_record["edges"].append(
                    (row["target_symbol_name"], row["relationship_type"])
                )
                current_record["edge_records"].append(
                    (row["edge_id"], row["target_symbol_name"], row["relationship_type"])
                )

        existing_graph = [(record["fields"], record["edges"]) for record in existing_records]
        if existing_graph != next_graph:
            existing_by_position: dict[tuple[str, int, int, int], dict[str, Any]] = {}
            existing_occurrences: dict[tuple[str, int, int], int] = {}
            for record in existing_records:
                fields = record["fields"]
                base = (fields[1], fields[4], fields[5])
                occurrence = existing_occurrences.get(base, 0)
                existing_occurrences[base] = occurrence + 1
                existing_by_position[(*base, occurrence)] = record

            matched_ids: set[int] = set()
            next_occurrences: dict[tuple[str, int, int], int] = {}
            updates: list[tuple[Any, ...]] = []
            edge_updates: list[tuple[str, str, int]] = []
            edge_deletions: list[tuple[int]] = []
            edge_rows: list[tuple[int, str, str]] = []
            new_symbols: list[tuple[CodeSymbol, list[SymbolEdge]]] = []

            for (symbol, edges), (fields, next_edges) in zip(
                symbols_with_edges, next_graph, strict=True
            ):
                base = (symbol.symbol_type, symbol.start_line, symbol.end_line)
                occurrence = next_occurrences.get(base, 0)
                next_occurrences[base] = occurrence + 1
                existing = existing_by_position.get((*base, occurrence))
                if existing is None:
                    new_symbols.append((symbol, edges))
                    continue

                symbol_id = existing["id"]
                matched_ids.add(symbol_id)
                if existing["fields"] != fields:
                    updates.append((*fields, symbol_id))
                if existing["edges"] != next_edges:
                    existing_edges = existing["edge_records"]
                    common_count = min(len(existing_edges), len(edges))
                    for index in range(common_count):
                        edge_id, target_name, relationship_type = existing_edges[index]
                        edge = edges[index]
                        if (target_name, relationship_type) != (
                            edge.target_symbol_name,
                            edge.relationship_type,
                        ):
                            edge_updates.append(
                                (edge.target_symbol_name, edge.relationship_type, edge_id)
                            )
                    edge_deletions.extend(
                        (edge_id,) for edge_id, _, _ in existing_edges[common_count:]
                    )
                    edge_rows.extend(
                        (symbol_id, edge.target_symbol_name, edge.relationship_type)
                        for edge in edges[common_count:]
                    )

            removed_ids = [
                (record["id"],) for record in existing_records if record["id"] not in matched_ids
            ]
            if len(removed_ids) == 1:
                conn.execute("DELETE FROM code_symbols WHERE id = ?", removed_ids[0])
            elif removed_ids:
                conn.executemany("DELETE FROM code_symbols WHERE id = ?", removed_ids)

            if updates:
                conn.executemany(
                    """
                    UPDATE code_symbols
                    SET symbol_name = ?, symbol_type = ?, signature = ?, docstring = ?,
                        start_line = ?, end_line = ?
                    WHERE id = ?
                    """,
                    updates,
                )

            if edge_updates:
                conn.executemany(
                    "UPDATE symbol_edges SET target_symbol_name = ?, relationship_type = ? WHERE id = ?",
                    edge_updates,
                )

            if edge_deletions:
                conn.executemany("DELETE FROM symbol_edges WHERE id = ?", edge_deletions)

            if new_symbols:
                conn.executemany(
                    """
                    INSERT INTO code_symbols (
                        project_key, file_path, symbol_name, symbol_type,
                        signature, docstring, start_line, end_line
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            project_key,
                            norm_path,
                            symbol.symbol_name,
                            symbol.symbol_type,
                            symbol.signature,
                            symbol.docstring,
                            symbol.start_line,
                            symbol.end_line,
                        )
                        for symbol, _ in new_symbols
                    ],
                )

                if any(edges for _, edges in new_symbols):
                    cursor = conn.execute(
                        """
                        SELECT id, symbol_name, symbol_type, signature, docstring,
                               start_line, end_line
                        FROM code_symbols
                        WHERE project_key = ? AND file_path = ?
                        ORDER BY id
                        """,
                        (project_key, norm_path),
                    )
                    new_ids_by_fields: dict[tuple[Any, ...], list[int]] = {}
                    for row in cursor.fetchall():
                        if row["id"] in matched_ids:
                            continue
                        fields = (
                            row["symbol_name"],
                            row["symbol_type"],
                            row["signature"],
                            row["docstring"],
                            row["start_line"],
                            row["end_line"],
                        )
                        new_ids_by_fields.setdefault(fields, []).append(row["id"])

                    for symbol, edges in new_symbols:
                        if not edges:
                            continue
                        fields = (
                            symbol.symbol_name,
                            symbol.symbol_type,
                            symbol.signature,
                            symbol.docstring,
                            symbol.start_line,
                            symbol.end_line,
                        )
                        symbol_ids = new_ids_by_fields.get(fields)
                        if not symbol_ids:
                            raise RuntimeError("Unable to resolve an inserted symbol ID.")
                        symbol_id = symbol_ids.pop(0)
                        edge_rows.extend(
                            (symbol_id, edge.target_symbol_name, edge.relationship_type)
                            for edge in edges
                        )

            if edge_rows:
                conn.executemany(
                    """
                    INSERT INTO symbol_edges (
                        source_symbol_id, target_symbol_name, relationship_type
                    ) VALUES (?, ?, ?)
                    """,
                    edge_rows,
                )

        graph_state_updates.append((project_key, norm_path, graph_digest))
        if mtime is not None:
            metadata_updates.append((project_key, norm_path, mtime))

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

    def _edge_neighbors(
        self,
        project_key: str,
        symbol: dict[str, Any],
        relationship_type: str,
        direction: str,
    ) -> list[dict[str, Any]]:
        if direction == "outgoing":
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT target_symbol_name FROM symbol_edges "
                    "WHERE source_symbol_id = ? AND relationship_type = ? "
                    "ORDER BY target_symbol_name",
                    (symbol.get("id"), relationship_type),
                )
                target_names = [row["target_symbol_name"] for row in cursor.fetchall()]

            neighbors: list[dict[str, Any]] = []
            for target_name in target_names:
                definitions = self.find_definition(project_key, target_name)
                if definitions:
                    neighbors.extend(definitions)
                else:
                    neighbors.append(
                        {
                            "symbol_name": target_name,
                            "file_path": None,
                            "start_line": None,
                        }
                    )
            return neighbors

        if direction != "incoming":
            raise ValueError("direction must be 'incoming' or 'outgoing'.")

        symbol_name = str(symbol.get("symbol_name") or "")
        names = sorted({symbol_name, symbol_name.rsplit(".", 1)[-1]} - {""})
        placeholders = ", ".join("?" for _ in names)
        with self._get_connection() as conn:
            cursor = conn.execute(
                f"""
                SELECT DISTINCT s.id, s.file_path, s.symbol_name, s.symbol_type,
                       s.signature, s.docstring, s.start_line, s.end_line
                FROM symbol_edges e
                JOIN code_symbols s ON e.source_symbol_id = s.id
                WHERE s.project_key = ? AND e.relationship_type = ?
                  AND e.target_symbol_name IN ({placeholders})
                ORDER BY s.file_path, s.start_line, s.symbol_name
                """,
                (project_key, relationship_type, *names),
            )
            return [dict(row) for row in cursor.fetchall()]

    @staticmethod
    def _walk_edges(
        store: "CodeGraphStore",
        project_key: str,
        roots: list[dict[str, Any]],
        relationship_type: str,
        direction: str,
        max_depth: int,
    ) -> list[dict[str, Any]]:
        queue = [(root, 0) for root in roots]
        seen = {
            (root.get("symbol_name"), root.get("file_path"), root.get("start_line"))
            for root in roots
        }
        result: list[dict[str, Any]] = []

        while queue:
            current, depth = queue.pop(0)
            if depth >= max_depth:
                continue
            neighbors = store._edge_neighbors(project_key, current, relationship_type, direction)
            for neighbor in neighbors:
                key = (
                    neighbor.get("symbol_name"),
                    neighbor.get("file_path"),
                    neighbor.get("start_line"),
                )
                if key in seen:
                    continue
                seen.add(key)
                item = {**neighbor, "depth": depth + 1}
                result.append(item)
                queue.append((neighbor, depth + 1))

        return sorted(
            result,
            key=lambda item: (
                item["depth"],
                item.get("file_path") or "",
                item.get("start_line") or 0,
                item.get("symbol_name") or "",
            ),
        )

    def get_symbol_call_tree(
        self, project_key: str, symbol_name: str, max_depth: int = 3
    ) -> dict[str, Any]:
        """Returns bounded upstream callers and downstream callees for indexed symbols."""
        definitions = self.find_definition(project_key, symbol_name)
        return {
            "symbol_name": symbol_name,
            "project_key": project_key,
            "definitions": definitions,
            "callers": self._walk_edges(
                self, project_key, definitions, "calls", "incoming", max_depth
            ),
            "callees": self._walk_edges(
                self, project_key, definitions, "calls", "outgoing", max_depth
            ),
            "max_depth": max_depth,
        }

    def get_type_hierarchy(
        self, project_key: str, symbol_name: str, max_depth: int = 3
    ) -> dict[str, Any]:
        """Returns bounded base types and derived implementations from inheritance edges."""
        definitions = self.find_definition(project_key, symbol_name)
        return {
            "symbol_name": symbol_name,
            "project_key": project_key,
            "definitions": definitions,
            "bases": self._walk_edges(
                self, project_key, definitions, "inherits", "outgoing", max_depth
            ),
            "derived": self._walk_edges(
                self, project_key, definitions, "inherits", "incoming", max_depth
            ),
            "max_depth": max_depth,
        }

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

    def _extract_calls(self, node: Node, cursor: QueryCursor) -> list[str]:
        ordered_matches = []
        for order, (_, captures) in enumerate(cursor.matches(node)):
            call = captures["call"][0]
            function = captures["function"][0]
            ordered_matches.append((call.start_byte, -call.end_byte, order, _node_text(function)))
        ordered_matches.sort()
        return [name for _, _, _, name in ordered_matches]

    def parse_python_source(
        self, code: str, file_path: str, project_key: str
    ) -> list[tuple[CodeSymbol, list[SymbolEdge]]]:
        """Parses Python source code into symbols and dependency edges."""
        tree = self.parser.parse(code.encode("utf-8"))
        call_cursor = QueryCursor(PY_CALL_QUERY)
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

                            mcalls = self._extract_calls(bchild, call_cursor)
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

                fcalls = self._extract_calls(child, call_cursor)
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

    def parse_polyglot_source(
        self, code: str, file_path: str, project_key: str, suffix: str
    ) -> list[tuple[CodeSymbol, list[SymbolEdge]]]:
        """
        Parses TypeScript/JavaScript (.ts, .tsx, .js, .jsx), Go (.go), and Rust (.rs)
        source files into structural symbols (classes, structs, interfaces, functions, methods)
        and dependency edges (calls, inherits, imports).
        """
        results: list[tuple[CodeSymbol, list[SymbolEdge]]] = []
        lines = code.splitlines()
        total_lines = len(lines)
        if total_lines == 0:
            return results

        def _find_block_end(start_idx: int) -> int:
            depth = 0
            opened = False
            for idx in range(start_idx, total_lines):
                for ch in lines[idx]:
                    if ch == "{":
                        depth += 1
                        opened = True
                    elif ch == "}":
                        depth -= 1
                        if opened and depth <= 0:
                            return idx + 1
            return min(total_lines, start_idx + 1)

        def _extract_leading_comment(line_idx: int) -> str:
            doc_lines: list[str] = []
            idx = line_idx - 1
            while idx >= 0:
                stripped = lines[idx].strip()
                if stripped.startswith(("//", "///", "*", "/**", "*/")):
                    cleaned = re.sub(r"^(/+\**|\*+/|\*+)\s*", "", stripped).strip()
                    if cleaned:
                        doc_lines.insert(0, cleaned)
                    idx -= 1
                else:
                    break
            return " ".join(doc_lines)

        def _extract_body_calls(start_line: int, end_line: int, own_name: str) -> list[SymbolEdge]:
            body_text = "\n".join(lines[max(0, start_line - 1) : min(total_lines, end_line)])
            call_matches = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", body_text)
            keywords = {
                "if",
                "for",
                "while",
                "switch",
                "catch",
                "return",
                "function",
                "func",
                "fn",
                "match",
                "sizeof",
                "new",
                "super",
                own_name,
            }
            edges: list[SymbolEdge] = []
            seen: set[str] = set()
            for target in call_matches:
                if target not in keywords and target not in seen:
                    seen.add(target)
                    edges.append(SymbolEdge(target_symbol_name=target, relationship_type="calls"))
            return edges

        if suffix in {".ts", ".tsx", ".js", ".jsx"}:
            class_re = re.compile(
                r"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?(class|interface)\s+([A-Za-z_][A-Za-z0-9_]*)(?:\s+(?:extends|implements)\s+([A-Za-z0-9_,\s<>]+))?"
            )
            func_re = re.compile(
                r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)(?:\s*:\s*([^{]+))?"
            )
            arrow_re = re.compile(
                r"^\s*(?:export\s+)?(?:const|let)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:async\s*)?\(([^)]*)\)\s*(?::\s*([^=]+))?=>"
            )
            method_re = re.compile(
                r"^\s*(?:public\s+|private\s+|protected\s+|static\s+|async\s+)*([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)(?:\s*:\s*([^{]+))?\s*\{"
            )

            current_class: tuple[str, int] | None = None
            for i, line in enumerate(lines):
                line_no = i + 1
                if current_class and line_no > current_class[1]:
                    current_class = None

                cm = class_re.match(line)
                if cm:
                    kind, cname, bases = cm.group(1), cm.group(2), cm.group(3)
                    end_l = _find_block_end(i)
                    doc = _extract_leading_comment(i)
                    edges: list[SymbolEdge] = []
                    if bases:
                        for b in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", bases):
                            edges.append(
                                SymbolEdge(target_symbol_name=b, relationship_type="inherits")
                            )
                    sig = line.strip().rstrip("{").strip()
                    results.append(
                        (
                            CodeSymbol(
                                symbol_name=cname,
                                symbol_type="class",
                                start_line=line_no,
                                end_line=end_l,
                                signature=sig,
                                docstring=doc,
                                file_path=file_path,
                                project_key=project_key,
                            ),
                            edges,
                        )
                    )
                    if kind == "class":
                        current_class = (cname, end_l)
                    continue

                fm = func_re.match(line) or arrow_re.match(line)
                if fm:
                    fname = fm.group(1)
                    end_l = _find_block_end(i)
                    doc = _extract_leading_comment(i)
                    sig = line.strip().rstrip("{").strip()
                    fedges = _extract_body_calls(line_no, end_l, fname)
                    results.append(
                        (
                            CodeSymbol(
                                symbol_name=fname,
                                symbol_type="function",
                                start_line=line_no,
                                end_line=end_l,
                                signature=sig,
                                docstring=doc,
                                file_path=file_path,
                                project_key=project_key,
                            ),
                            fedges,
                        )
                    )
                    continue

                if current_class:
                    mm = method_re.match(line)
                    if mm:
                        mname = mm.group(1)
                        if mname not in {"if", "for", "while", "switch", "catch"}:
                            end_l = _find_block_end(i)
                            doc = _extract_leading_comment(i)
                            sig = line.strip().rstrip("{").strip()
                            medges = _extract_body_calls(line_no, end_l, mname)
                            results.append(
                                (
                                    CodeSymbol(
                                        symbol_name=f"{current_class[0]}.{mname}",
                                        symbol_type="method",
                                        start_line=line_no,
                                        end_line=end_l,
                                        signature=sig,
                                        docstring=doc,
                                        file_path=file_path,
                                        project_key=project_key,
                                    ),
                                    medges,
                                )
                            )

        elif suffix == ".go":
            type_re = re.compile(r"^\s*type\s+([A-Za-z_][A-Za-z0-9_]*)\s+(struct|interface)\b")
            func_re = re.compile(
                r"^\s*func\s+(?:\(\s*\w+\s+\*?([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*\("
            )
            for i, line in enumerate(lines):
                line_no = i + 1
                tm = type_re.match(line)
                if tm:
                    tname = tm.group(1)
                    end_l = _find_block_end(i)
                    doc = _extract_leading_comment(i)
                    sig = line.strip().rstrip("{").strip()
                    results.append(
                        (
                            CodeSymbol(
                                symbol_name=tname,
                                symbol_type="class",
                                start_line=line_no,
                                end_line=end_l,
                                signature=sig,
                                docstring=doc,
                                file_path=file_path,
                                project_key=project_key,
                            ),
                            [],
                        )
                    )
                    continue
                fm = func_re.match(line)
                if fm:
                    recv, fname = fm.group(1), fm.group(2)
                    end_l = _find_block_end(i)
                    doc = _extract_leading_comment(i)
                    sig = line.strip().rstrip("{").strip()
                    fedges = _extract_body_calls(line_no, end_l, fname)
                    sym_name = f"{recv}.{fname}" if recv else fname
                    sym_type = "method" if recv else "function"
                    results.append(
                        (
                            CodeSymbol(
                                symbol_name=sym_name,
                                symbol_type=sym_type,
                                start_line=line_no,
                                end_line=end_l,
                                signature=sig,
                                docstring=doc,
                                file_path=file_path,
                                project_key=project_key,
                            ),
                            fedges,
                        )
                    )

        elif suffix == ".rs":
            type_re = re.compile(
                r"^\s*(?:pub(?:\([^)]*\))?\s+)?(struct|enum|trait)\s+([A-Za-z_][A-Za-z0-9_]*)"
            )
            impl_re = re.compile(
                r"^\s*impl(?:<[^>]*>)?\s+(?:([A-Za-z_][A-Za-z0-9_]*)\s+for\s+)?([A-Za-z_][A-Za-z0-9_]*)"
            )
            fn_re = re.compile(
                r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*[<(]"
            )
            current_impl: tuple[str, int] | None = None
            for i, line in enumerate(lines):
                line_no = i + 1
                if current_impl and line_no > current_impl[1]:
                    current_impl = None

                tm = type_re.match(line)
                if tm:
                    tname = tm.group(2)
                    end_l = _find_block_end(i)
                    doc = _extract_leading_comment(i)
                    sig = line.strip().rstrip("{").strip()
                    results.append(
                        (
                            CodeSymbol(
                                symbol_name=tname,
                                symbol_type="class",
                                start_line=line_no,
                                end_line=end_l,
                                signature=sig,
                                docstring=doc,
                                file_path=file_path,
                                project_key=project_key,
                            ),
                            [],
                        )
                    )
                    continue

                im = impl_re.match(line)
                if im:
                    trait_name, struct_name = im.group(1), im.group(2)
                    end_l = _find_block_end(i)
                    current_impl = (struct_name, end_l)
                    if trait_name:
                        results.append(
                            (
                                CodeSymbol(
                                    symbol_name=struct_name,
                                    symbol_type="class",
                                    start_line=line_no,
                                    end_line=end_l,
                                    signature=line.strip().rstrip("{").strip(),
                                    file_path=file_path,
                                    project_key=project_key,
                                ),
                                [
                                    SymbolEdge(
                                        target_symbol_name=trait_name, relationship_type="inherits"
                                    )
                                ],
                            )
                        )
                    continue

                fm = fn_re.match(line)
                if fm:
                    fname = fm.group(1)
                    end_l = _find_block_end(i)
                    doc = _extract_leading_comment(i)
                    sig = line.strip().rstrip("{").strip()
                    fedges = _extract_body_calls(line_no, end_l, fname)
                    sym_name = f"{current_impl[0]}.{fname}" if current_impl else fname
                    sym_type = "method" if current_impl else "function"
                    results.append(
                        (
                            CodeSymbol(
                                symbol_name=sym_name,
                                symbol_type=sym_type,
                                start_line=line_no,
                                end_line=end_l,
                                signature=sig,
                                docstring=doc,
                                file_path=file_path,
                                project_key=project_key,
                            ),
                            fedges,
                        )
                    )

        return results

    def _parse_file(
        self, file_path: Path | str, project_key: str | None
    ) -> tuple[str, str, list[tuple[CodeSymbol, list[SymbolEdge]]], float] | None:
        path = Path(file_path).resolve()
        if not path.is_file() or path.suffix not in SUPPORTED_EXTENSIONS:
            return None

        p_key = project_key or resolve_project_key()
        try:
            st_mtime = path.stat().st_mtime
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None

        rel_path = str(path)
        if path.suffix == ".py":
            symbols_with_edges = self.parse_python_source(content, rel_path, p_key)
        else:
            symbols_with_edges = self.parse_polyglot_source(content, rel_path, p_key, path.suffix)
        return p_key, rel_path, symbols_with_edges, st_mtime

    def index_file(self, file_path: Path | str, project_key: str | None = None) -> int:
        """Indexes one supported source file into the code graph store."""
        parsed = self._parse_file(file_path, project_key)
        if parsed is None:
            return 0
        p_key, rel_path, symbols_with_edges, st_mtime = parsed
        self.store.save_file_symbols(p_key, rel_path, symbols_with_edges, mtime=st_mtime)
        return len(symbols_with_edges)

    def index_files(self, file_paths: list[Path | str], project_key: str | None = None) -> int:
        """Indexes several source files and persists their graphs in one transaction."""
        parsed_files = [
            parsed
            for file_path in file_paths
            if (parsed := self._parse_file(file_path, project_key)) is not None
        ]
        if not parsed_files:
            return 0
        self.store.save_files_symbols(parsed_files)
        return sum(len(symbols_with_edges) for _, _, symbols_with_edges, _ in parsed_files)

    def index_directory(
        self,
        dir_path: Path | str,
        project_key: str | None = None,
        max_files: int = 500,
    ) -> int:
        """Recursively indexes all supported source files under dir_path, skipping unchanged files via st_mtime."""
        root = Path(dir_path).resolve()
        if not root.is_dir():
            return 0

        p_key = project_key or resolve_project_key()
        indexed_count = 0

        for current_root, dirs, files in os.walk(root):
            # Prune excluded directories in-place
            dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS and not d.startswith(".")]

            for file in files:
                if Path(file).suffix in SUPPORTED_EXTENSIONS:
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
