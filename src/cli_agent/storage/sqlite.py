"""Standard-library SQLite storage engine with WAL and bounded lock waiting."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from .base import StorageConnection, quote_identifier


class SQLiteEngine:
    fts_dialect = "fts5"
    migration_transaction_sql = "BEGIN IMMEDIATE"

    def __init__(self, database_path: Path | str, busy_timeout_ms: int = 30_000) -> None:
        self.database_path = Path(database_path).expanduser()
        self.busy_timeout_ms = max(1, int(busy_timeout_ms))
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def local_database_path(self) -> Path:
        return self.database_path

    def connect(self) -> StorageConnection:
        connection = sqlite3.connect(
            str(self.database_path),
            timeout=self.busy_timeout_ms / 1000,
            check_same_thread=False,
        )
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms}")
        connection.execute("PRAGMA journal_mode = WAL")
        return StorageConnection(connection)

    def execute(self, sql: str, parameters: Sequence[Any] = ()) -> list[dict[str, Any]]:
        with self.connect() as connection:
            cursor = connection.execute(sql, parameters)
            if cursor.description is None:
                return []
            return [dict(row) for row in cursor.fetchall()]

    def executemany(self, sql: str, parameters: Iterable[Sequence[Any]]) -> int:
        with self.connect() as connection:
            return connection.executemany(sql, parameters).rowcount

    def execute_transaction(self, statements: Iterable[tuple[str, Sequence[Any]]]) -> None:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for sql, parameters in statements:
                connection.execute(sql, parameters)

    def create_fts_table(
        self,
        table_name: str,
        columns: Sequence[str],
        *,
        content_table: str | None = None,
        content_rowid: str | None = None,
    ) -> None:
        statement = self.fts_table_statement(
            table_name,
            columns,
            content_table=content_table,
            content_rowid=content_rowid,
        )
        self.execute(statement)

    def fts_table_statement(
        self,
        table_name: str,
        columns: Sequence[str],
        *,
        content_table: str | None = None,
        content_rowid: str | None = None,
    ) -> str:
        return self._fts_statement(
            table_name,
            columns,
            content_table=content_table,
            content_rowid=content_rowid,
        )

    @staticmethod
    def _fts_statement(
        table_name: str,
        columns: Sequence[str],
        *,
        content_table: str | None = None,
        content_rowid: str | None = None,
    ) -> str:
        if not columns:
            raise ValueError("An FTS table requires at least one column.")
        quoted_columns = ", ".join(quote_identifier(column) for column in columns)
        options = ""
        if content_table is not None:
            content_table = quote_identifier(content_table)[1:-1]
            options += f", content='{content_table}'"
        if content_rowid is not None:
            if content_table is None:
                raise ValueError("content_rowid requires content_table.")
            content_rowid = quote_identifier(content_rowid)[1:-1]
            options += f", content_rowid='{content_rowid}'"
        return (
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {quote_identifier(table_name)} "
            f"USING fts5({quoted_columns}{options})"
        )

    def close(self) -> None:
        """SQLite connections are scoped and closed at the end of each operation."""
