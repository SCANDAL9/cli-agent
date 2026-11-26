"""Tests for unified storage backends and lossless legacy migration."""

import hashlib
import sqlite3
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from cli_agent.code_graph import CodeGraphStore
from cli_agent.config import StorageConfig, load_storage_config
from cli_agent.memory import MemoryStore
from cli_agent.search import HybridSearchEngine
from cli_agent.session import SessionStore
from cli_agent.storage import (
    SQLiteEngine,
    StorageBackendUnavailable,
    StorageMigrationConflict,
    TursoEngine,
    clear_storage_engine_cache,
    create_storage_engine,
    get_storage_engine,
    migrate_legacy_database,
    resolve_store_engine,
)


@pytest.mark.unit
def test_sqlite_engine_executes_rows_many_and_transactions(tmp_path: Path):
    engine = SQLiteEngine(tmp_path / "store.db")
    engine.execute("CREATE TABLE records (id INTEGER PRIMARY KEY, value TEXT NOT NULL)")

    engine.executemany("INSERT INTO records (id, value) VALUES (?, ?)", [(1, "alpha"), (2, "beta")])
    assert engine.execute("SELECT id, value FROM records ORDER BY id") == [
        {"id": 1, "value": "alpha"},
        {"id": 2, "value": "beta"},
    ]

    with pytest.raises(sqlite3.IntegrityError):
        engine.execute_transaction(
            [
                ("INSERT INTO records (id, value) VALUES (?, ?)", (3, "gamma")),
                ("INSERT INTO records (id, value) VALUES (?, ?)", (1, "duplicate")),
            ]
        )

    assert engine.execute("SELECT id FROM records ORDER BY id") == [{"id": 1}, {"id": 2}]
    engine.close()


@pytest.mark.unit
def test_sqlite_engine_creates_parameterized_fts5_tables(tmp_path: Path):
    engine = SQLiteEngine(tmp_path / "fts.db")
    engine.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY, title TEXT NOT NULL)")
    engine.create_fts_table(
        "documents_fts", ["title"], content_table="documents", content_rowid="id"
    )
    engine.execute("INSERT INTO documents VALUES (?, ?)", (1, "storage engine"))
    engine.execute("INSERT INTO documents_fts(documents_fts) VALUES ('rebuild')")

    assert engine.execute(
        "SELECT rowid FROM documents_fts WHERE documents_fts MATCH ?", ("storage",)
    ) == [{"rowid": 1}]
    with pytest.raises(ValueError, match="Invalid SQL identifier"):
        engine.create_fts_table("documents; DROP TABLE documents", ["title"])
    engine.close()


@pytest.mark.unit
def test_turso_engine_executes_common_storage_operations_and_syncs_replica(
    monkeypatch, tmp_path: Path
):
    class FakeLibSQLConnection:
        def __init__(self):
            self.raw_connection = sqlite3.connect(":memory:")
            self.sync_calls = 0
            self.fail_sync = False

        def cursor(self):
            return self.raw_connection.cursor()

        def execute(self, sql, parameters=()):
            return self.raw_connection.execute(sql, parameters)

        def executemany(self, sql, parameters):
            return self.raw_connection.executemany(sql, parameters)

        def commit(self):
            self.raw_connection.commit()

        def rollback(self):
            self.raw_connection.rollback()

        def sync(self):
            self.sync_calls += 1
            if self.fail_sync:
                raise RuntimeError("replica unavailable")

        def close(self):
            self.raw_connection.close()

    fake_connection = FakeLibSQLConnection()
    fake_module = ModuleType("libsql")
    fake_module.connect = lambda *_args, **_kwargs: fake_connection
    monkeypatch.setitem(sys.modules, "libsql", fake_module)

    engine = TursoEngine(tmp_path / "replica.db", "libsql://db.example", "test-token")
    assert (
        engine.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY, title TEXT NOT NULL)") == []
    )
    assert (
        engine.executemany(
            "INSERT INTO documents (id, title) VALUES (?, ?)", [(1, "storage"), (2, "engine")]
        )
        == 2
    )
    with pytest.raises(sqlite3.IntegrityError):
        engine.execute_transaction(
            [
                ("INSERT INTO documents (id, title) VALUES (?, ?)", (3, "rolled back")),
                ("INSERT INTO documents (id, title) VALUES (?, ?)", (1, "duplicate")),
            ]
        )
    assert engine.execute("SELECT id FROM documents ORDER BY id") == [{"id": 1}, {"id": 2}]
    engine.execute_transaction(
        [("UPDATE documents SET title = ? WHERE id = ?", ("storage engine", 1))]
    )
    engine.create_fts_table(
        "documents_fts", ["title"], content_table="documents", content_rowid="id"
    )
    engine.execute("INSERT INTO documents_fts(documents_fts) VALUES ('rebuild')")

    assert engine.execute(
        "SELECT rowid FROM documents_fts WHERE documents_fts MATCH ?", ("storage",)
    ) == [{"rowid": 1}]
    assert fake_connection.sync_calls == 1
    engine.sync()
    assert fake_connection.sync_calls == 2
    fake_connection.fail_sync = True
    with pytest.raises(StorageBackendUnavailable, match="replica synchronization failed"):
        engine.sync()
    engine.close()


@pytest.mark.unit
def test_turso_engine_remote_mode_skips_replica_sync(monkeypatch, tmp_path: Path):
    calls: list[tuple[str, dict[str, Any]]] = []

    class FakeRemoteConnection:
        def __init__(self):
            self.raw_connection = sqlite3.connect(":memory:")

        def cursor(self):
            return self.raw_connection.cursor()

        def execute(self, sql, parameters=()):
            return self.raw_connection.execute(sql, parameters)

        def commit(self):
            self.raw_connection.commit()

        def rollback(self):
            self.raw_connection.rollback()

        def close(self):
            self.raw_connection.close()

    fake_connection = FakeRemoteConnection()
    fake_module = ModuleType("libsql")

    def fake_connect(database: str, **kwargs: Any):
        calls.append((database, kwargs))
        return fake_connection

    fake_module.connect = fake_connect
    monkeypatch.setitem(sys.modules, "libsql", fake_module)
    engine = TursoEngine(
        tmp_path / "unused-replica.db",
        "libsql://db.example",
        "test-token",
        mode="remote",
    )

    assert engine.local_database_path is None
    assert engine.execute("CREATE TABLE remote_records (id INTEGER PRIMARY KEY)") == []
    engine.sync()

    assert engine.database_path == "libsql://db.example"
    assert calls == [("libsql://db.example", {"auth_token": "test-token"})]
    engine.close()


@pytest.mark.unit
@pytest.mark.parametrize("failure", ["missing_driver", "connection"])
def test_turso_engine_reports_unavailable_backend(monkeypatch, tmp_path: Path, failure: str):
    fake_module = ModuleType("libsql")
    if failure == "missing_driver":

        def missing_driver(_name: str):
            raise ImportError("libsql is not installed")

        monkeypatch.setattr("cli_agent.storage.turso.import_module", missing_driver)
        message = "requires the optional 'libsql' dependency"
    else:

        def fail_connection(*_args: Any, **_kwargs: Any):
            raise OSError("connection unavailable")

        fake_module.connect = fail_connection
        monkeypatch.setitem(sys.modules, "libsql", fake_module)
        message = "Unable to initialize the Turso embedded replica"

    engine = TursoEngine(tmp_path / "replica.db", "libsql://db.example", "test-token")

    with pytest.raises(StorageBackendUnavailable, match=message):
        engine.connect()

    engine.close()


@pytest.mark.unit
def test_turso_engine_supports_real_libsql_local_db_api(tmp_path: Path):
    pytest.importorskip("libsql")
    database_path = tmp_path / "local-libsql.db"
    engine = TursoEngine(database_path, str(database_path), "local-test-token", mode="remote")

    assert engine.execute("CREATE TABLE records (id INTEGER PRIMARY KEY, value TEXT)") == []
    assert engine.executemany("INSERT INTO records (id, value) VALUES (?, ?)", [(1, "alpha")]) == 1
    assert engine.execute("SELECT id, value FROM records") == [{"id": 1, "value": "alpha"}]
    engine.close()


@pytest.mark.unit
def test_storage_factory_defaults_to_sqlite_without_turso_credentials(tmp_path: Path):
    engine = create_storage_engine(tmp_path / "memory.db", StorageConfig())

    assert isinstance(engine, SQLiteEngine)


@pytest.mark.unit
@pytest.mark.parametrize("backend", ["turso", "auto"])
def test_storage_factory_migrates_legacy_records_into_replica_without_changing_source(
    monkeypatch, tmp_path: Path, backend: str
):
    legacy_path = tmp_path / "legacy.db"
    source = SQLiteEngine(legacy_path)
    source.execute("CREATE TABLE records (id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
    source.execute("INSERT INTO records (id, value) VALUES (?, ?)", (1, "keep me"))
    source.close()
    original_hash = hashlib.sha256(legacy_path.read_bytes()).hexdigest()
    calls: list[tuple[str, dict[str, Any]]] = []

    class FakeConnection:
        def __init__(self):
            self.connection = sqlite3.connect(":memory:")
            self.sync_calls = 0

        def sync(self):
            self.sync_calls += 1

        def cursor(self):
            return self.connection.cursor()

        def execute(self, sql, parameters=()):
            return self.connection.execute(sql, parameters)

        def commit(self):
            self.connection.commit()

        def rollback(self):
            self.connection.rollback()

        def close(self):
            self.connection.close()

    fake_connection = FakeConnection()
    fake_module = ModuleType("libsql")

    def fake_connect(database, **kwargs):
        calls.append((database, kwargs))
        return fake_connection

    fake_module.connect = fake_connect
    monkeypatch.setitem(sys.modules, "libsql", fake_module)
    replica_path = tmp_path / "replica.db"
    engine = create_storage_engine(
        legacy_path,
        StorageConfig(
            backend=backend,
            turso_database_url="libsql://db.example",
            turso_auth_token="test-token",
            replica_path=replica_path,
        ),
    )

    assert engine.execute("SELECT id, value FROM records") == [{"id": 1, "value": "keep me"}]
    assert hashlib.sha256(legacy_path.read_bytes()).hexdigest() == original_hash
    assert fake_connection.sync_calls == 2
    assert calls == [
        (
            str(replica_path),
            {
                "sync_url": "libsql://db.example",
                "auth_token": "test-token",
                "sync_interval": 30.0,
            },
        )
    ]
    engine.close()


@pytest.mark.unit
def test_storage_factory_skips_migration_when_replica_is_the_legacy_database(
    monkeypatch, tmp_path: Path
):
    legacy_path = tmp_path / "memory.db"
    source = SQLiteEngine(legacy_path)
    source.execute("CREATE TABLE records (id INTEGER PRIMARY KEY)")
    source.close()
    migration_calls: list[tuple[object, object]] = []
    monkeypatch.setattr(
        "cli_agent.storage.factory.migrate_legacy_database",
        lambda source_path, destination: migration_calls.append((source_path, destination)),
    )

    engine = create_storage_engine(
        legacy_path,
        StorageConfig(
            backend="turso",
            turso_database_url="libsql://db.example",
            turso_auth_token="test-token",
            replica_path=legacy_path,
        ),
    )

    assert isinstance(engine, TursoEngine)
    assert engine.local_database_path == legacy_path
    assert migration_calls == []
    engine.close()


@pytest.mark.unit
def test_default_engine_cache_reuses_sqlite_engine_for_matching_settings(tmp_path: Path):
    path = tmp_path / "shared.db"
    first = get_storage_engine(path, StorageConfig(backend="sqlite"))
    second = get_storage_engine(path, StorageConfig(backend="sqlite"))

    assert first is second
    clear_storage_engine_cache()


@pytest.mark.unit
def test_store_engine_resolution_prefers_explicit_engine_and_honors_explicit_config(
    tmp_path: Path,
):
    default_path = tmp_path / "default.db"
    custom_path = tmp_path / "custom.db"
    explicit_engine = SQLiteEngine(tmp_path / "explicit.db")
    settings = StorageConfig(backend="sqlite")

    assert (
        resolve_store_engine(
            default_path,
            database_path=custom_path,
            storage_engine=explicit_engine,
            storage_config=settings,
        )
        is explicit_engine
    )

    configured_engine = resolve_store_engine(
        default_path,
        database_path=custom_path,
        storage_config=settings,
    )
    assert configured_engine is get_storage_engine(custom_path, settings)
    assert isinstance(configured_engine, SQLiteEngine)
    clear_storage_engine_cache()
    explicit_engine.close()


@pytest.mark.unit
def test_storage_factory_rejects_partial_turso_configuration(tmp_path: Path):
    config = StorageConfig(backend="auto", turso_database_url="libsql://db.example")

    with pytest.raises(ValueError, match="both TURSO_DATABASE_URL and TURSO_AUTH_TOKEN"):
        create_storage_engine(tmp_path / "memory.db", config)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("config", "message"),
    [
        (StorageConfig(backend="unknown"), "Storage backend must be"),
        (StorageConfig(backend="turso"), "requires both TURSO_DATABASE_URL and TURSO_AUTH_TOKEN"),
    ],
)
def test_storage_factory_rejects_invalid_backend_or_missing_turso_credentials(
    tmp_path: Path, config: StorageConfig, message: str
):
    with pytest.raises(ValueError, match=message):
        create_storage_engine(tmp_path / "memory.db", config)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("variable", "value", "message"),
    [
        ("CLI_AGENT_STORAGE_BACKEND", "unknown", "CLI_AGENT_STORAGE_BACKEND must be"),
        ("TURSO_MODE", "unsupported", "TURSO_MODE must be"),
        ("TURSO_SYNC_INTERVAL", "not-a-number", "TURSO_SYNC_INTERVAL must be a positive"),
        ("TURSO_SYNC_INTERVAL", "0", "TURSO_SYNC_INTERVAL must be a positive"),
    ],
)
def test_storage_config_rejects_invalid_environment_settings(
    monkeypatch, tmp_path: Path, variable: str, value: str, message: str
):
    for name in (
        "CLI_AGENT_STORAGE_BACKEND",
        "TURSO_MODE",
        "TURSO_SYNC_INTERVAL",
        "TURSO_DATABASE_URL",
        "TURSO_AUTH_TOKEN",
        "TURSO_REPLICA_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(variable, value)

    with pytest.raises(ValueError, match=message):
        load_storage_config(tmp_path / "missing-config.toml")


@pytest.mark.unit
def test_storage_config_keeps_auth_token_environment_only_and_hidden(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[storage]\nbackend = "turso"\ndatabase_url = "libsql://db.example"\n'
        'auth_token = "must-not-be-read"\nsync_interval = 12\n',
        encoding="utf-8",
    )
    for name in (
        "CLI_AGENT_STORAGE_BACKEND",
        "TURSO_DATABASE_URL",
        "TURSO_MODE",
        "TURSO_SYNC_INTERVAL",
        "TURSO_REPLICA_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TURSO_AUTH_TOKEN", "test-token")

    config = load_storage_config(config_path)

    assert config.backend == "turso"
    assert config.turso_database_url == "libsql://db.example"
    assert config.turso_auth_token == "test-token"
    assert config.turso_mode == "replica"
    assert config.sync_interval == 12
    assert "test-token" not in repr(config)


@pytest.mark.unit
def test_storage_config_accepts_remote_only_turso_mode(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("TURSO_MODE", "remote")
    monkeypatch.delenv("TURSO_SYNC_INTERVAL", raising=False)

    config = load_storage_config(tmp_path / "missing-config.toml")

    assert config.turso_mode == "remote"


@pytest.mark.unit
def test_turso_engine_uses_embedded_replica_and_adapts_rows(monkeypatch, tmp_path: Path):
    calls = []

    class FakeConnection:
        def __init__(self):
            self.connection = sqlite3.connect(":memory:")
            self.sync_calls = 0

        def sync(self):
            self.sync_calls += 1

        def cursor(self):
            return self.connection.cursor()

        def execute(self, sql, parameters=()):
            return self.connection.execute(sql, parameters)

        def commit(self):
            self.connection.commit()

        def rollback(self):
            self.connection.rollback()

        def close(self):
            self.connection.close()

    fake_connection = FakeConnection()
    fake_module = ModuleType("libsql")

    def fake_connect(database, **kwargs):
        calls.append((database, kwargs))
        return fake_connection

    fake_module.connect = fake_connect
    monkeypatch.setitem(sys.modules, "libsql", fake_module)
    engine = TursoEngine(
        tmp_path / "replica.db",
        remote_url="libsql://db.example",
        auth_token="test-token",
        sync_interval=12,
    )

    with engine.connect() as connection:
        row = connection.execute("SELECT 7 AS value").fetchone()

    assert row["value"] == 7
    assert row[0] == 7
    assert fake_connection.sync_calls == 1
    assert calls == [
        (
            str(tmp_path / "replica.db"),
            {
                "sync_url": "libsql://db.example",
                "auth_token": "test-token",
                "sync_interval": 12.0,
            },
        )
    ]
    engine.close()


@pytest.mark.unit
def test_turso_engine_uses_remote_only_connection_without_replica_sync(monkeypatch, tmp_path: Path):
    calls = []

    class FakeConnection:
        def __init__(self):
            self.connection = sqlite3.connect(":memory:")
            self.sync_calls = 0

        def sync(self):
            self.sync_calls += 1

        def cursor(self):
            return self.connection.cursor()

        def execute(self, sql, parameters=()):
            return self.connection.execute(sql, parameters)

        def commit(self):
            self.connection.commit()

        def rollback(self):
            self.connection.rollback()

        def close(self):
            self.connection.close()

    fake_connection = FakeConnection()
    fake_module = ModuleType("libsql")

    def fake_connect(database, **kwargs):
        calls.append((database, kwargs))
        return fake_connection

    fake_module.connect = fake_connect
    monkeypatch.setitem(sys.modules, "libsql", fake_module)
    engine = TursoEngine(
        tmp_path / "unused-replica.db",
        remote_url="libsql://db.example",
        auth_token="test-token",
        mode="remote",
    )

    with engine.connect() as connection:
        row = connection.execute("SELECT 9 AS value").fetchone()
    engine.sync()

    assert row["value"] == 9
    assert engine.database_path == "libsql://db.example"
    assert engine.local_database_path is None
    assert fake_connection.sync_calls == 0
    assert calls == [("libsql://db.example", {"auth_token": "test-token"})]
    engine.close()


@pytest.mark.unit
def test_turso_engine_fails_closed_when_optional_driver_is_unavailable(monkeypatch, tmp_path: Path):
    monkeypatch.setitem(sys.modules, "libsql", None)
    engine = TursoEngine(
        tmp_path / "replica.db",
        remote_url="libsql://db.example",
        auth_token="test-token",
    )

    with pytest.raises(StorageBackendUnavailable, match="optional 'libsql' dependency"):
        engine.connect()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("remote_url", "auth_token", "sync_interval", "mode", "message"),
    [
        ("", "test-token", 30.0, "replica", "requires both"),
        ("libsql://db.example", "", 30.0, "replica", "requires both"),
        ("libsql://db.example", "test-token", 0.0, "replica", "must be positive"),
        ("libsql://db.example", "test-token", 30.0, "unsupported", "mode must be"),
    ],
)
def test_turso_engine_rejects_invalid_connection_configuration(
    tmp_path: Path,
    remote_url: str,
    auth_token: str,
    sync_interval: float,
    mode: str,
    message: str,
):
    with pytest.raises(ValueError, match=message):
        TursoEngine(
            tmp_path / "replica.db",
            remote_url=remote_url,
            auth_token=auth_token,
            sync_interval=sync_interval,
            mode=mode,
        )


@pytest.mark.unit
def test_turso_engine_normalizes_connection_initialization_errors(monkeypatch, tmp_path: Path):
    fake_module = ModuleType("libsql")

    def fail_connect(*_args, **_kwargs):
        raise OSError("driver connection failed")

    fake_module.connect = fail_connect
    monkeypatch.setitem(sys.modules, "libsql", fake_module)
    engine = TursoEngine(
        tmp_path / "replica.db",
        remote_url="libsql://db.example",
        auth_token="test-token",
    )

    with pytest.raises(StorageBackendUnavailable, match="Unable to initialize"):
        engine.connect()

    engine.close()


@pytest.mark.unit
def test_turso_engine_reports_replica_sync_failures(monkeypatch, tmp_path: Path):
    class FakeConnection:
        def __init__(self):
            self.connection = sqlite3.connect(":memory:")

        def sync(self):
            return None

        def cursor(self):
            return self.connection.cursor()

        def execute(self, sql, parameters=()):
            return self.connection.execute(sql, parameters)

        def commit(self):
            self.connection.commit()

        def rollback(self):
            self.connection.rollback()

        def close(self):
            self.connection.close()

    fake_connection = FakeConnection()
    fake_module = ModuleType("libsql")
    fake_module.connect = lambda *_args, **_kwargs: fake_connection
    monkeypatch.setitem(sys.modules, "libsql", fake_module)
    engine = TursoEngine(
        tmp_path / "replica.db",
        remote_url="libsql://db.example",
        auth_token="test-token",
    )
    engine.connect()

    def fail_sync():
        raise OSError("replica unavailable")

    fake_connection.sync = fail_sync
    with pytest.raises(StorageBackendUnavailable, match="replica synchronization failed"):
        engine.sync()

    engine.close()


@pytest.mark.unit
def test_all_persistent_stores_share_the_default_engine_for_a_custom_database(tmp_path: Path):
    database_path = tmp_path / "shared.db"
    stores = [
        MemoryStore(database_path),
        SessionStore(database_path),
        CodeGraphStore(database_path),
        HybridSearchEngine(database_path),
    ]
    engine = stores[0].storage_engine

    assert all(store.storage_engine is engine for store in stores)
    table_names = {
        row["name"] for row in engine.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    assert {
        "memories",
        "sessions",
        "code_symbols",
        "symbol_edges",
        "file_metadata",
        "code_chunks",
        "memories_fts",
        "code_chunks_fts",
    } <= table_names
    clear_storage_engine_cache()


def _create_legacy_database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE memories (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE memories_fts USING fts5(
                title,
                content,
                content='memories',
                content_rowid='id'
            );
            CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, title, content)
                VALUES (new.id, new.title, new.content);
            END;
            CREATE TABLE sessions (id TEXT PRIMARY KEY, messages_json TEXT NOT NULL);
            CREATE TABLE code_symbols (id INTEGER PRIMARY KEY, symbol_name TEXT NOT NULL);
            CREATE TABLE code_chunks (id INTEGER PRIMARY KEY, content TEXT NOT NULL);
            INSERT INTO memories VALUES (17, 'Needle', 'searchable memory');
            INSERT INTO sessions VALUES ('sess-1', '[{"role":"user"}]');
            INSERT INTO code_symbols VALUES (23, 'build_index');
            INSERT INTO code_chunks VALUES (31, 'graph-indexed chunk');
            """
        )


@pytest.mark.unit
def test_legacy_migration_preserves_all_tables_and_is_idempotent(tmp_path: Path):
    source = tmp_path / "memory.db"
    target = tmp_path / "active.db"
    _create_legacy_database(source)
    engine = SQLiteEngine(target)

    first = migrate_legacy_database(source, engine)
    second = migrate_legacy_database(source, engine)

    assert first.migrated is True
    assert set(first.copied_tables) >= {
        "memories",
        "sessions",
        "code_symbols",
        "code_chunks",
    }
    assert second.already_migrated is True
    assert engine.execute("SELECT id, title FROM memories") == [{"id": 17, "title": "Needle"}]
    assert engine.execute("SELECT id FROM sessions") == [{"id": "sess-1"}]
    assert engine.execute("SELECT id FROM code_symbols") == [{"id": 23}]
    assert engine.execute("SELECT id FROM code_chunks") == [{"id": 31}]
    assert engine.execute(
        "SELECT rowid FROM memories_fts WHERE memories_fts MATCH ?", ("Needle",)
    ) == [{"rowid": 17}]
    engine.execute("INSERT INTO memories VALUES (?, ?, ?)", (18, "Future", "triggered index"))
    assert engine.execute(
        "SELECT rowid FROM memories_fts WHERE memories_fts MATCH ?", ("Future",)
    ) == [{"rowid": 18}]
    with sqlite3.connect(source) as connection:
        assert connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1
    engine.close()


@pytest.mark.unit
def test_legacy_migration_conflict_rolls_back_without_changing_source(tmp_path: Path):
    source = tmp_path / "legacy.db"
    target = tmp_path / "active.db"
    with sqlite3.connect(source) as connection:
        connection.executescript(
            """
            CREATE TABLE memories (id INTEGER PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE sessions (id TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO memories VALUES (1, 'legacy');
            INSERT INTO sessions VALUES ('legacy-session', 'from source');
            """
        )
    engine = SQLiteEngine(target)
    engine.execute("CREATE TABLE memories (id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
    engine.execute("INSERT INTO memories VALUES (?, ?)", (1, "different target value"))

    with pytest.raises(StorageMigrationConflict, match="memories"):
        migrate_legacy_database(source, engine)

    assert engine.execute("SELECT * FROM memories") == [
        {"id": 1, "value": "different target value"}
    ]
    assert (
        engine.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'sessions'")
        == []
    )
    with sqlite3.connect(source) as connection:
        assert connection.execute("SELECT value FROM memories").fetchone()[0] == "legacy"
        assert connection.execute("SELECT value FROM sessions").fetchone()[0] == "from source"
    engine.close()


@pytest.mark.unit
def test_legacy_migration_accepts_an_identical_existing_primary_key_row(tmp_path: Path):
    source = tmp_path / "legacy.db"
    target = tmp_path / "active.db"
    with sqlite3.connect(source) as connection:
        connection.execute("CREATE TABLE memories (id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute("INSERT INTO memories VALUES (?, ?)", (1, "same value"))

    engine = SQLiteEngine(target)
    engine.execute("CREATE TABLE memories (id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
    engine.execute("INSERT INTO memories VALUES (?, ?)", (1, "same value"))

    report = migrate_legacy_database(source, engine)

    assert report.migrated is True
    assert engine.execute("SELECT * FROM memories") == [{"id": 1, "value": "same value"}]
    with sqlite3.connect(source) as connection:
        assert connection.execute("SELECT * FROM memories").fetchall() == [(1, "same value")]
    engine.close()


@pytest.mark.unit
def test_legacy_migration_rejects_incompatible_existing_schema(tmp_path: Path):
    source = tmp_path / "legacy.db"
    target = tmp_path / "active.db"
    with sqlite3.connect(source) as connection:
        connection.executescript(
            """
            CREATE TABLE memories (id INTEGER PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO memories VALUES (1, 'legacy');
            """
        )
    engine = SQLiteEngine(target)
    engine.execute("CREATE TABLE memories (id INTEGER PRIMARY KEY, content BLOB NOT NULL)")

    with pytest.raises(StorageMigrationConflict, match="Incompatible schema.*memories"):
        migrate_legacy_database(source, engine)

    assert (
        engine.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'sessions'")
        == []
    )
    assert engine.execute("SELECT content FROM memories") == []
    engine.close()
