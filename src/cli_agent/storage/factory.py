"""Storage backend selection and shared default-engine lifecycle."""

from __future__ import annotations

import threading
from pathlib import Path

from ..config import StorageConfig, load_storage_config
from .base import StorageEngine
from .migration import migrate_legacy_database
from .sqlite import SQLiteEngine
from .turso import TursoEngine, TursoLocalEngine


def _turso_database_path(legacy_path: Path, config: StorageConfig) -> Path:
    if config.replica_path is not None:
        return config.replica_path
    suffix = legacy_path.suffix or ".db"
    return legacy_path.with_name(f"{legacy_path.stem}.turso{suffix}")


def _migrate_or_close(legacy_path: Path, engine: StorageEngine) -> None:
    try:
        migrate_legacy_database(legacy_path, engine)
    except BaseException:
        engine.close()
        raise


def create_storage_engine(
    database_path: Path | str,
    config: StorageConfig,
) -> StorageEngine:
    """Builds the configured backend, importing the legacy SQLite DB into Turso."""
    legacy_path = Path(database_path).expanduser()
    has_url = bool(config.turso_database_url)
    has_token = bool(config.turso_auth_token)
    backend = config.backend

    if backend == "auto":
        if has_url != has_token:
            raise ValueError("auto storage requires both TURSO_DATABASE_URL and TURSO_AUTH_TOKEN")
        backend = "turso"
    if backend == "sqlite":
        return SQLiteEngine(legacy_path, synchronous=config.synchronous)
    if backend != "turso":
        raise ValueError("Storage backend must be 'auto', 'sqlite', or 'turso'.")
    if not has_url and not has_token:
        if config.turso_mode == "remote":
            raise ValueError(
                "remote Turso storage requires both TURSO_DATABASE_URL and TURSO_AUTH_TOKEN"
            )
        engine = TursoLocalEngine(
            _turso_database_path(legacy_path, config), synchronous=config.synchronous
        )
        if legacy_path.resolve() != engine.local_database_path.resolve():
            _migrate_or_close(legacy_path, engine)
        return engine
    if not has_url or not has_token:
        raise ValueError("turso storage requires both TURSO_DATABASE_URL and TURSO_AUTH_TOKEN")

    if config.turso_mode == "remote":
        engine = TursoEngine(
            str(config.turso_database_url),
            remote_url=str(config.turso_database_url),
            auth_token=str(config.turso_auth_token),
            sync_interval=config.sync_interval,
            mode="remote",
            synchronous=config.synchronous,
        )
    else:
        replica_path = _turso_database_path(legacy_path, config)
        engine = TursoEngine(
            replica_path,
            remote_url=str(config.turso_database_url),
            auth_token=str(config.turso_auth_token),
            sync_interval=config.sync_interval,
            synchronous=config.synchronous,
        )
    local_path = engine.local_database_path
    if local_path is None or legacy_path.resolve() != local_path.resolve():
        _migrate_or_close(legacy_path, engine)
    return engine


_ENGINE_CACHE: dict[tuple[Path, StorageConfig], StorageEngine] = {}
_ENGINE_CACHE_LOCK = threading.RLock()


def get_storage_engine(
    database_path: Path | str,
    config: StorageConfig | None = None,
) -> StorageEngine:
    """Returns one engine shared by the default stores using the same settings."""
    settings = config or load_storage_config()
    key = (Path(database_path).expanduser().resolve(), settings)
    with _ENGINE_CACHE_LOCK:
        if key not in _ENGINE_CACHE:
            _ENGINE_CACHE[key] = create_storage_engine(key[0], settings)
        return _ENGINE_CACHE[key]


def resolve_store_engine(
    default_database_path: Path,
    database_path: Path | str | None = None,
    storage_engine: StorageEngine | None = None,
    storage_config: StorageConfig | None = None,
) -> StorageEngine:
    """Preserves explicit custom database paths as SQLite unless otherwise configured."""
    if storage_engine is not None:
        return storage_engine
    path = Path(database_path).expanduser() if database_path is not None else default_database_path
    if storage_config is not None:
        settings = storage_config
    else:
        settings = load_storage_config()
    return get_storage_engine(path, settings)


def clear_storage_engine_cache() -> None:
    """Closes cached engines and clears them, primarily for application shutdown/tests."""
    with _ENGINE_CACHE_LOCK:
        for engine in _ENGINE_CACHE.values():
            engine.close()
        _ENGINE_CACHE.clear()
