#!/usr/bin/env python3
"""Compare durable single-file code-graph updates on local Turso and SQLite.

Run from the repository root with:
    PYTHONPATH=src .venv/bin/python scripts/benchmark_code_graph_incremental.py

All databases and generated source files live in a temporary directory. The
probe uses identical workloads per backend and reports candidate/source hashes
so measurements can be tied to the exact code that was tested.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from cli_agent.code_graph import CodeGraphIndexer, CodeGraphStore
from cli_agent.storage.sqlite import SQLiteEngine
from cli_agent.storage.turso import TursoLocalEngine

PROJECT_KEY = "incremental-benchmark"
INITIAL_FILES = 500
SYMBOLS_PER_FILE = 50
MODIFICATIONS_PER_TRIAL = 40
CREATES_PER_TRIAL = 20
TRIALS = 3


def _source(file_number: int, call_target: int | None = None) -> str:
    lines = [
        f"def function_{file_number}_{index}(): return {index}" for index in range(SYMBOLS_PER_FILE)
    ]
    if call_target is not None:
        lines[0] = f"def function_{file_number}_0(): return helper_{call_target}()"
    return "\n".join(lines) + "\n"


def _new_file_source(file_number: int, call_target: int) -> str:
    return f"def created_{file_number}(): return helper_{call_target}()\n"


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    rank = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[rank]


def _summary(values: list[float]) -> dict[str, float | int]:
    return {
        "samples": len(values),
        "p50_ms": round(statistics.median(values), 3),
        "p95_ms": round(_percentile(values, 0.95), 3),
    }


def _candidate_identity(repository_root: Path) -> dict[str, str | None]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        tracked_diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"head": None, "tracked_diff_sha256": None, "code_graph_sha256": None}

    graph_path = repository_root / "src" / "cli_agent" / "code_graph.py"
    benchmark_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    graph_hash = hashlib.sha256(graph_path.read_bytes()).hexdigest()
    return {
        "head": head,
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "code_graph_sha256": graph_hash,
        "benchmark_script_sha256": benchmark_hash,
    }


def _counts(store: CodeGraphStore) -> tuple[int, int]:
    with store._get_connection() as connection:
        symbols = connection.execute(
            "SELECT COUNT(*) AS count FROM code_symbols WHERE project_key = ?",
            (PROJECT_KEY,),
        ).fetchone()
        edges = connection.execute("SELECT COUNT(*) AS count FROM symbol_edges").fetchone()
    assert symbols is not None and edges is not None
    return symbols["count"], edges["count"]


def _run_trial(root: Path, backend: str, trial: int) -> dict[str, Any]:
    trial_root = root / backend / f"trial_{trial}"
    source_root = trial_root / "project"
    source_root.mkdir(parents=True)
    initial_paths = [source_root / f"file_{index:04}.py" for index in range(INITIAL_FILES)]
    for index, path in enumerate(initial_paths):
        path.write_text(_source(index), encoding="utf-8")

    database_path = trial_root / "graph.db"
    engine = TursoLocalEngine(database_path) if backend == "turso" else SQLiteEngine(database_path)
    store = CodeGraphStore(database_path, storage_engine=engine)
    indexer = CodeGraphIndexer(store)
    indexer.index_files(initial_paths, PROJECT_KEY)
    assert _counts(store) == (25_000, 0)

    parse_ms: list[float] = []
    persistence_ms: list[float] = []
    event_ms: dict[str, list[float]] = {"modify": [], "create": []}
    original_parse = indexer._parse_file
    original_save = store.save_file_symbols

    def measured_parse(
        file_path: Path | str, project_key: str | None
    ) -> tuple[str, str, list[Any], float] | None:
        start = time.perf_counter()
        try:
            return original_parse(file_path, project_key)
        finally:
            parse_ms.append((time.perf_counter() - start) * 1000)

    def measured_save(
        project_key: str,
        file_path: str,
        symbols_with_edges: list[Any],
        mtime: float | None = None,
    ) -> None:
        start = time.perf_counter()
        try:
            original_save(project_key, file_path, symbols_with_edges, mtime)
        finally:
            persistence_ms.append((time.perf_counter() - start) * 1000)

    indexer._parse_file = measured_parse
    store.save_file_symbols = measured_save

    for index in range(MODIFICATIONS_PER_TRIAL):
        file_path = source_root / f"file_{index:04}.py"
        file_path.write_text(_source(index, call_target=trial * 1_000 + index), encoding="utf-8")
        start = time.perf_counter()
        indexer.index_file(file_path, PROJECT_KEY)
        event_ms["modify"].append((time.perf_counter() - start) * 1000)

    for index in range(CREATES_PER_TRIAL):
        file_number = INITIAL_FILES + index
        file_path = source_root / f"created_{file_number:04}.py"
        file_path.write_text(
            _new_file_source(file_number, trial * 1_000 + file_number),
            encoding="utf-8",
        )
        start = time.perf_counter()
        indexer.index_file(file_path, PROJECT_KEY)
        event_ms["create"].append((time.perf_counter() - start) * 1000)

    final_counts = _counts(store)
    assert final_counts == (25_020, 60), final_counts
    engine.close()
    return {
        "trial": trial,
        "events": {kind: _summary(values) for kind, values in event_ms.items()},
        "stages": {
            "parse_read": _summary(parse_ms),
            "persist_transaction": _summary(persistence_ms),
        },
        "final_symbols": final_counts[0],
        "final_edges": final_counts[1],
        "_samples": event_ms,
        "_parse_ms": parse_ms,
        "_persistence_ms": persistence_ms,
    }


def _run_backend(root: Path, backend: str) -> dict[str, Any]:
    trials = [_run_trial(root, backend, trial) for trial in range(1, TRIALS + 1)]
    aggregate = {
        "events": {
            kind: _summary([sample for trial in trials for sample in trial["_samples"][kind]])
            for kind in ("modify", "create")
        },
        "stages": {
            "parse_read": _summary([sample for trial in trials for sample in trial["_parse_ms"]]),
            "persist_transaction": _summary(
                [sample for trial in trials for sample in trial["_persistence_ms"]]
            ),
        },
    }
    for trial in trials:
        del trial["_samples"]
        del trial["_parse_ms"]
        del trial["_persistence_ms"]
    return {"backend": backend, "aggregate": aggregate, "trials": trials}


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    candidate = _candidate_identity(repository_root)
    with tempfile.TemporaryDirectory(prefix="cli-agent-code-graph-benchmark-") as temp:
        results = [
            _run_backend(Path(temp), "turso"),
            _run_backend(Path(temp), "sqlite"),
        ]
    print(
        json.dumps(
            {
                "candidate": candidate,
                "environment": {
                    "python": sys.version.split()[0],
                    "platform": platform.platform(),
                },
                "workload": {
                    "trials": TRIALS,
                    "initial_files": INITIAL_FILES,
                    "symbols_per_file": SYMBOLS_PER_FILE,
                    "initial_symbols": 25_000,
                    "initial_edges": 0,
                    "modifications_per_trial": MODIFICATIONS_PER_TRIAL,
                    "creates_per_trial": CREATES_PER_TRIAL,
                    "expected_final_symbols_per_trial": 25_020,
                    "expected_final_edges_per_trial": 60,
                    "percentile_method": "nearest-rank",
                },
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
