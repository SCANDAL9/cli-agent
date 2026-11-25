"""Optional libSQL embedded-replica storage engine for Turso."""

from __future__ import annotations

import threading
from collections.abc import Iterable, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any

from .base import StorageBackendUnavailable, StorageConnection


class TursoEngine:
    """Keeps a local libSQL replica and synchronizes it with the configured remote."""

    def __init__(
        self,
        database_path: Path | str,
        remote_url: str,
        auth_token: str,
        sync_interval: float = 30.0,
        mode: str = "replica",
    ) -> None:
        if not remote_url or not auth_token:
            raise ValueError("Turso requires both a database URL and an authentication token.")
        if sync_interval <= 0:
            raise ValueError("Turso sync interval must be positive.")
        if mode not in {"replica", "remote"}:
            raise ValueError("Turso mode must be 'replica' or 'remote'.")
        self.mode = mode
        self._local_database_path: Path | None = None
        if mode == "replica":
            local_database_path = Path(database_path).expanduser()
            local_database_path.parent.mkdir(parents=True, exist_ok=True)
            self._local_database_path = local_database_path
            self.database_path: Path | str = local_database_path
        else:
            self.database_path = remote_url
        self.remote_url = remote_url
        self._auth_token = auth_token
        self.sync_interval = float(sync_interval)
        self._connection: Any | None = None
        self._lock = threading.RLock()

    @property
    def local_database_path(self) -> Path | None:
        return self._local_database_path

    def _open_connection(self) -> Any:
        try:
            libsql = import_module("libsql")
        except ImportError as exc:
            raise StorageBackendUnavailable(
                "Turso storage requires the optional 'libsql' dependency; install cli-agent[turso]."
            ) from exc

        try:
            if self.mode == "remote":
                connection = libsql.connect(self.remote_url, auth_token=self._auth_token)
            else:
                connection = libsql.connect(
                    str(self.database_path),
                    sync_url=self.remote_url,
                    auth_token=self._auth_token,
                    sync_interval=self.sync_interval,
                )
                connection.sync()
            connection.execute("PRAGMA foreign_keys = ON")
        except Exception as exc:
            raise StorageBackendUnavailable(
                "Unable to initialize the Turso embedded replica."
            ) from exc
        return connection

    def connect(self) -> StorageConnection:
        with self._lock:
            if self._connection is None:
                self._connection = self._open_connection()
            connection = self._connection
        return StorageConnection(connection, close_on_exit=False, lock=self._lock)

    def execute(self, sql: str, parameters: Sequence[Any] = ()) -> list[dict[str, Any]]:
        with self.connect() as connection:
            cursor = connection.execute(sql, parameters)
            if not cursor.description:
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
        from .sqlite import SQLiteEngine

        # DDL generation is shared; execution stays on the replica connection.
        statement = SQLiteEngine._fts_statement(
            table_name,
            columns,
            content_table=content_table,
            content_rowid=content_rowid,
        )
        self.execute(statement)

    def sync(self) -> None:
        if self.mode == "remote":
            return
        with self.connect() as connection:
            try:
                connection._connection.sync()
            except Exception as exc:
                raise StorageBackendUnavailable("Turso replica synchronization failed.") from exc

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                try:
                    self._connection.close()
                finally:
                    self._connection = None
