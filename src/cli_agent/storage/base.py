"""DB-API normalization and common storage engine contracts."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol, cast, runtime_checkable


class StorageBackendUnavailable(RuntimeError):
    """Raised when a configured storage backend cannot be initialized."""


class StorageRow(Mapping[str, Any]):
    """Mapping row compatible with sqlite3.Row's name and positional access."""

    def __init__(self, columns: Sequence[str], values: Sequence[Any]) -> None:
        self._columns = tuple(columns)
        self._values = tuple(values)
        self._mapping = dict(zip(self._columns, self._values, strict=True))

    def __getitem__(self, key: str | int) -> Any:
        if isinstance(key, int):
            return self._values[key]
        return self._mapping[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._columns)

    def __len__(self) -> int:
        return len(self._columns)


def _translate_db_error(error: Exception) -> Exception:
    if isinstance(error, sqlite3.Error):
        return error
    translated = getattr(sqlite3, error.__class__.__name__, None)
    if isinstance(translated, type) and issubclass(translated, sqlite3.Error):
        return translated(str(error))
    return error


class StorageCursor:
    """Small cursor proxy that converts tuple rows into mapping-compatible rows."""

    def __init__(self, cursor: Any) -> None:
        self._cursor = cursor

    @property
    def description(self) -> Any:
        return self._cursor.description

    @property
    def lastrowid(self) -> int | None:
        return cast(int | None, self._cursor.lastrowid)

    @property
    def rowcount(self) -> int:
        return cast(int, self._cursor.rowcount)

    def execute(self, sql: str, parameters: Sequence[Any] = ()) -> StorageCursor:
        try:
            self._cursor.execute(sql, parameters)
        except Exception as exc:
            raise _translate_db_error(exc) from exc
        return self

    def executemany(self, sql: str, parameters: Iterable[Sequence[Any]]) -> StorageCursor:
        try:
            self._cursor.executemany(sql, parameters)
        except Exception as exc:
            raise _translate_db_error(exc) from exc
        return self

    def _convert(self, row: Any) -> StorageRow | None:
        if row is None:
            return None
        description = self._cursor.description or ()
        columns = tuple(item[0] for item in description)
        return StorageRow(columns, row)

    def fetchone(self) -> Any:
        try:
            return self._convert(self._cursor.fetchone())
        except Exception as exc:
            raise _translate_db_error(exc) from exc

    def fetchall(self) -> list[StorageRow]:
        try:
            converted: list[StorageRow] = []
            for row in self._cursor.fetchall():
                item = self._convert(row)
                if item is not None:
                    converted.append(item)
            return converted
        except Exception as exc:
            raise _translate_db_error(exc) from exc

    def __iter__(self) -> Iterator[StorageRow]:
        while True:
            row = self.fetchone()
            if row is None:
                return
            yield row


class StorageConnection:
    """DB-API connection proxy with consistent commit, rollback, and row behavior."""

    def __init__(
        self,
        connection: Any,
        *,
        close_on_exit: bool = True,
        lock: Any | None = None,
    ) -> None:
        self._connection = connection
        self._close_on_exit = close_on_exit
        self._lock = lock

    def __enter__(self) -> StorageConnection:
        if self._lock is not None:
            self._lock.acquire()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> Literal[False]:
        try:
            if exc_type is None:
                self.commit()
            else:
                self.rollback()
            if self._close_on_exit:
                self.close()
        finally:
            if self._lock is not None:
                self._lock.release()
        return False

    def cursor(self) -> StorageCursor:
        try:
            return StorageCursor(self._connection.cursor())
        except Exception as exc:
            raise _translate_db_error(exc) from exc

    def execute(self, sql: str, parameters: Sequence[Any] = ()) -> StorageCursor:
        cursor = self.cursor()
        return cursor.execute(sql, parameters)

    def executemany(self, sql: str, parameters: Iterable[Sequence[Any]]) -> StorageCursor:
        cursor = self.cursor()
        return cursor.executemany(sql, parameters)

    def commit(self) -> None:
        try:
            self._connection.commit()
        except Exception as exc:
            raise _translate_db_error(exc) from exc

    def rollback(self) -> None:
        try:
            self._connection.rollback()
        except Exception as exc:
            raise _translate_db_error(exc) from exc

    def close(self) -> None:
        try:
            self._connection.close()
        except Exception as exc:
            raise _translate_db_error(exc) from exc


@runtime_checkable
class StorageEngine(Protocol):
    """Common storage interface used by memory, session, graph, and search stores."""

    @property
    def database_path(self) -> Path | str: ...

    @property
    def local_database_path(self) -> Path | None: ...

    def connect(self) -> StorageConnection: ...

    def execute(self, sql: str, parameters: Sequence[Any] = ()) -> list[dict[str, Any]]: ...

    def executemany(self, sql: str, parameters: Iterable[Sequence[Any]]) -> int: ...

    def execute_transaction(self, statements: Iterable[tuple[str, Sequence[Any]]]) -> None: ...

    def create_fts_table(
        self,
        table_name: str,
        columns: Sequence[str],
        *,
        content_table: str | None = None,
        content_rowid: str | None = None,
    ) -> None: ...

    def close(self) -> None: ...


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def quote_identifier(identifier: str) -> str:
    if not _IDENTIFIER.fullmatch(identifier):
        raise ValueError(f"Invalid SQL identifier: {identifier!r}")
    return f'"{identifier}"'
