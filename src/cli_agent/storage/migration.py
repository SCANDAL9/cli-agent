"""Lossless, transactional migration of a legacy SQLite database."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import StorageEngine, quote_identifier

MIGRATION_ID = "legacy_memory_database_v1"


class StorageMigrationConflict(RuntimeError):
    """Raised when the active database contains a conflicting legacy record/schema."""


@dataclass(frozen=True)
class MigrationReport:
    migrated: bool = False
    already_migrated: bool = False
    source_missing: bool = False
    copied_tables: tuple[str, ...] = ()
    copied_rows: int = 0


def _with_if_not_exists(statement: str, object_type: str) -> str:
    match = re.match(
        rf"^(CREATE\s+(?:VIRTUAL\s+)?{object_type}\s+)(?!IF\s+NOT\s+EXISTS\b)", statement, re.I
    )
    if match:
        return statement[: match.end()] + "IF NOT EXISTS " + statement[match.end() :]
    return statement


def _read_only_connection(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)


def _table_kinds(source: sqlite3.Connection) -> dict[str, str]:
    rows = source.execute("PRAGMA table_list").fetchall()
    return {str(row[1]): str(row[2]) for row in rows}


def _columns(source: sqlite3.Connection, table_name: str) -> list[str]:
    rows = source.execute(f"PRAGMA table_xinfo({quote_identifier(table_name)})").fetchall()
    return [str(row[1]) for row in rows if int(row[6]) == 0]


def _primary_key(source: sqlite3.Connection, table_name: str) -> list[str]:
    rows = source.execute(f"PRAGMA table_info({quote_identifier(table_name)})").fetchall()
    return [str(row[1]) for row in sorted(rows, key=lambda row: int(row[5])) if int(row[5])]


def _legacy_fts_definition(
    source: sqlite3.Connection,
    table_name: str,
    statement: str,
) -> tuple[list[str], str, str]:
    if not re.search(r"\bUSING\s+fts5\s*\(", statement, re.I):
        raise StorageMigrationConflict(
            f"Unsupported legacy full-text index '{table_name}'; source data was not modified."
        )
    content = re.search(r"\bcontent\s*=\s*(['\"])(.*?)\1", statement, re.I)
    if content is None or not content.group(2):
        raise StorageMigrationConflict(
            f"Legacy full-text index '{table_name}' has no rebuildable content table."
        )
    content_rowid = re.search(r"\bcontent_rowid\s*=\s*(['\"])(.*?)\1", statement, re.I)
    columns = _columns(source, table_name)
    if not columns:
        raise StorageMigrationConflict(f"Legacy full-text index '{table_name}' has no columns.")
    return columns, content.group(2), content_rowid.group(2) if content_rowid else "rowid"


def _validate_target_schema(source: sqlite3.Connection, destination: Any, table_name: str) -> None:
    quoted_table = quote_identifier(table_name)
    source_columns = source.execute(f"PRAGMA table_xinfo({quoted_table})").fetchall()
    destination_columns = destination.execute(f"PRAGMA table_xinfo({quoted_table})").fetchall()
    destination_by_name = {str(row["name"]): row for row in destination_columns}
    for column in source_columns:
        name = str(column[1])
        existing = destination_by_name.get(name)
        if existing is None or (
            str(existing["type"] or "").upper() != str(column[2] or "").upper()
            or int(existing["pk"]) != int(column[5])
            or int(existing["hidden"]) != int(column[6])
        ):
            raise StorageMigrationConflict(f"Incompatible schema for legacy table '{table_name}'.")


def _copy_table_rows(
    source: sqlite3.Connection,
    destination: Any,
    table_name: str,
) -> int:
    columns = _columns(source, table_name)
    if not columns:
        return 0
    quoted_table = quote_identifier(table_name)
    quoted_columns = ", ".join(quote_identifier(column) for column in columns)
    source_cursor = source.execute(f"SELECT {quoted_columns} FROM {quoted_table}")
    placeholders = ", ".join("?" for _ in columns)
    insert_sql = f"INSERT INTO {quoted_table} ({quoted_columns}) VALUES ({placeholders})"
    primary_key = _primary_key(source, table_name)
    copied = 0

    for raw_row in source_cursor:
        values = tuple(raw_row)
        if primary_key:
            key_positions = [columns.index(column) for column in primary_key]
            key_values = tuple(values[index] for index in key_positions)
            where_clause = " AND ".join(f"{quote_identifier(column)} = ?" for column in primary_key)
            existing = destination.execute(
                f"SELECT {quoted_columns} FROM {quoted_table} WHERE {where_clause}", key_values
            ).fetchone()
            if existing is not None:
                existing_values = tuple(existing[column] for column in columns)
                if existing_values != values:
                    raise StorageMigrationConflict(
                        f"Conflicting primary-key row in legacy table '{table_name}'."
                    )
                continue
        destination.execute(insert_sql, values)
        copied += 1
    return copied


def migrate_legacy_database(source_path: Path | str, engine: StorageEngine) -> MigrationReport:
    """Copies every legacy table and its searchable data without modifying the source."""
    source_path = Path(source_path).expanduser()
    if not source_path.is_file():
        return MigrationReport(source_missing=True)
    destination_path = engine.local_database_path
    if destination_path is not None and source_path.resolve() == destination_path.resolve():
        return MigrationReport(already_migrated=True)

    with _read_only_connection(source_path) as source:
        kinds = _table_kinds(source)
        objects = source.execute(
            "SELECT type, name, sql FROM sqlite_master "
            "WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%' "
            "AND type IN ('table', 'index', 'trigger', 'view') "
            "ORDER BY CASE type WHEN 'table' THEN 0 WHEN 'index' THEN 2 "
            "WHEN 'trigger' THEN 3 ELSE 4 END, name"
        ).fetchall()
        table_objects = [row for row in objects if row[0] == "table"]
        other_objects = [row for row in objects if row[0] != "table"]
        user_tables = [str(row[1]) for row in table_objects if kinds.get(str(row[1])) != "shadow"]
        native_fts = getattr(engine, "fts_dialect", "fts5") == "turso"
        fts_indexes: dict[str, tuple[list[str], str, str]] = {}
        if native_fts:
            for row in table_objects:
                name, statement = str(row[1]), str(row[2])
                if kinds.get(name) == "virtual":
                    fts_indexes[name] = _legacy_fts_definition(source, name, statement)

        with engine.connect() as destination:
            destination.execute(
                "CREATE TABLE IF NOT EXISTS __cli_agent_storage_migrations "
                "(migration_id TEXT PRIMARY KEY, source_path TEXT NOT NULL, completed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )
            marker = destination.execute(
                "SELECT migration_id FROM __cli_agent_storage_migrations WHERE migration_id = ?",
                (MIGRATION_ID,),
            ).fetchone()
            if marker:
                return MigrationReport(already_migrated=True, copied_tables=tuple(user_tables))

            destination.execute(getattr(engine, "migration_transaction_sql", "BEGIN IMMEDIATE"))
            for row in table_objects:
                name, statement = str(row[1]), str(row[2])
                if kinds.get(name) == "shadow" or name in fts_indexes:
                    continue
                destination.execute(_with_if_not_exists(statement, "TABLE"))
                _validate_target_schema(source, destination, name)

            copied_rows = 0
            for table_name in user_tables:
                kind = kinds.get(table_name, "table")
                row = next(row for row in table_objects if row[1] == table_name)
                statement = str(row[2])
                if table_name in fts_indexes:
                    continue
                if kind == "virtual" and "content=" in statement.lower():
                    continue
                copied_rows += _copy_table_rows(source, destination, table_name)

            for row in other_objects:
                kind, statement = str(row[0]), str(row[2])
                if (
                    native_fts
                    and kind == "trigger"
                    and any(
                        re.search(rf"\b{re.escape(name)}\b", statement, re.I)
                        for name in fts_indexes
                    )
                ):
                    continue
                if (
                    native_fts
                    and kind == "view"
                    and any(
                        re.search(rf"\b{re.escape(name)}\b", statement, re.I)
                        for name in fts_indexes
                    )
                ):
                    raise StorageMigrationConflict(
                        "A legacy view depends on an FTS5 virtual table and cannot be translated safely."
                    )
                if kind == "index":
                    statement = _with_if_not_exists(statement, "INDEX")
                elif kind == "trigger":
                    statement = _with_if_not_exists(statement, "TRIGGER")
                elif kind == "view":
                    statement = _with_if_not_exists(statement, "VIEW")
                destination.execute(statement)

            for name, (columns, content_table, content_rowid) in fts_indexes.items():
                destination.execute(
                    engine.fts_table_statement(
                        name,
                        columns,
                        content_table=content_table,
                        content_rowid=content_rowid,
                    )
                )

            if not native_fts:
                for row in table_objects:
                    name, statement = str(row[1]), str(row[2])
                    if kinds.get(name) == "virtual" and "content=" in statement.lower():
                        quoted = quote_identifier(name)
                        destination.execute(f"INSERT INTO {quoted}({quoted}) VALUES ('rebuild')")

            destination.execute(
                "INSERT INTO __cli_agent_storage_migrations (migration_id, source_path) VALUES (?, ?)",
                (MIGRATION_ID, str(source_path.resolve())),
            )

    sync = getattr(engine, "sync", None)
    if callable(sync):
        sync()
    return MigrationReport(
        migrated=True,
        copied_tables=tuple(user_tables),
        copied_rows=copied_rows,
    )
