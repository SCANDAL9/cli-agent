"""Helpers for selecting and verifying per-connection durability modes."""

from __future__ import annotations

from typing import Any

_SYNCHRONOUS_MODES = {"OFF": 0, "NORMAL": 1, "FULL": 2}


def normalize_synchronous_mode(mode: str) -> str:
    normalized = mode.strip().upper()
    if normalized not in _SYNCHRONOUS_MODES:
        raise ValueError("Synchronous mode must be 'OFF', 'NORMAL', or 'FULL'.")
    return normalized


def configure_synchronous(connection: Any, mode: str) -> None:
    normalized = normalize_synchronous_mode(mode)
    expected = _SYNCHRONOUS_MODES[normalized]
    connection.execute(f"PRAGMA synchronous = {normalized}")
    row = connection.execute("PRAGMA synchronous").fetchone()
    if row is None or int(row[0]) != expected:
        actual = None if row is None else row[0]
        raise RuntimeError(
            f"Database rejected synchronous mode {normalized}; read back {actual!r}."
        )
