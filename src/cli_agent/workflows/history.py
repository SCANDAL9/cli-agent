"""Durable workflow run and step history stored through StorageEngine."""

from __future__ import annotations

import json
from typing import Any

from ..storage import StorageEngine
from ..tools import _redact_secrets


class WorkflowHistoryStore:
    """Persists workflow state without binding the runner to a database backend."""

    def __init__(self, storage_engine: StorageEngine) -> None:
        self.storage_engine = storage_engine
        with self.storage_engine.connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS workflow_runs (
                    run_id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    inputs_json TEXT NOT NULL,
                    branch_name TEXT,
                    tag_name TEXT,
                    worktree_path TEXT,
                    error TEXT,
                    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    finished_at TEXT
                )"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS workflow_steps (
                    run_id TEXT NOT NULL REFERENCES workflow_runs(run_id) ON DELETE CASCADE,
                    step_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempt INTEGER NOT NULL DEFAULT 1,
                    output_json TEXT,
                    exit_code INTEGER,
                    error TEXT,
                    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    finished_at TEXT,
                    PRIMARY KEY (run_id, step_id)
                )"""
            )
            run_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(workflow_runs)").fetchall()
            }
            if "tag_name" not in run_columns:
                connection.execute("ALTER TABLE workflow_runs ADD COLUMN tag_name TEXT")

    @staticmethod
    def _encode(value: Any) -> str:
        serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        return _redact_secrets(serialized)

    @staticmethod
    def _safe_text(value: str | None) -> str | None:
        return _redact_secrets(value) if value is not None else None

    def start_run(self, run_id: str, workflow_id: str, inputs: dict[str, Any]) -> None:
        self.storage_engine.execute(
            "INSERT INTO workflow_runs (run_id, workflow_id, status, inputs_json) VALUES (?, ?, ?, ?)",
            (run_id, workflow_id, "starting", self._encode(inputs)),
        )

    def set_checkpoint(
        self, run_id: str, branch_name: str, tag_name: str, worktree_path: str
    ) -> None:
        self.storage_engine.execute(
            "UPDATE workflow_runs SET branch_name = ?, tag_name = ?, worktree_path = ? WHERE run_id = ?",
            (branch_name, tag_name, worktree_path, run_id),
        )

    def update_step(
        self,
        run_id: str,
        step_id: str,
        *,
        status: str,
        attempt: int = 1,
        output: Any = None,
        exit_code: int | None = None,
        error: str | None = None,
        finished: bool = False,
    ) -> None:
        finished_at = "CURRENT_TIMESTAMP" if finished else "NULL"
        self.storage_engine.execute(
            """INSERT INTO workflow_steps
               (run_id, step_id, status, attempt, output_json, exit_code, error, finished_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, """
            + finished_at
            + """)
               ON CONFLICT(run_id, step_id) DO UPDATE SET
                 status = excluded.status, attempt = excluded.attempt,
                 output_json = excluded.output_json, exit_code = excluded.exit_code,
                 error = excluded.error,
                 finished_at = CASE WHEN excluded.finished_at IS NULL
                   THEN workflow_steps.finished_at ELSE excluded.finished_at END""",
            (
                run_id,
                step_id,
                status,
                attempt,
                self._encode(output) if output is not None else None,
                exit_code,
                self._safe_text(error),
            ),
        )

    def finish_run(
        self,
        run_id: str,
        *,
        status: str,
        error: str | None = None,
        keep_checkpoint: bool = False,
    ) -> None:
        if not keep_checkpoint:
            self.storage_engine.execute(
                "UPDATE workflow_runs SET status = ?, error = ?, worktree_path = NULL, finished_at = CURRENT_TIMESTAMP WHERE run_id = ?",
                (status, self._safe_text(error), run_id),
            )
        else:
            self.storage_engine.execute(
                "UPDATE workflow_runs SET status = ?, error = ?, finished_at = CURRENT_TIMESTAMP WHERE run_id = ?",
                (status, self._safe_text(error), run_id),
            )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        rows = self.storage_engine.execute(
            "SELECT * FROM workflow_runs WHERE run_id = ?", (run_id,)
        )
        if not rows:
            return None
        result = rows[0]
        result["inputs"] = json.loads(result.pop("inputs_json"))
        result["steps"] = self.list_steps(run_id)
        return result

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        bounded_limit = min(max(int(limit), 1), 100)
        runs = self.storage_engine.execute(
            "SELECT * FROM workflow_runs ORDER BY started_at DESC LIMIT ?", (bounded_limit,)
        )
        for run in runs:
            run["inputs"] = json.loads(run.pop("inputs_json"))
        return runs

    def list_steps(self, run_id: str) -> list[dict[str, Any]]:
        steps = self.storage_engine.execute(
            "SELECT * FROM workflow_steps WHERE run_id = ? ORDER BY started_at, step_id",
            (run_id,),
        )
        for step in steps:
            raw_output = step.pop("output_json")
            step["output"] = json.loads(raw_output) if raw_output is not None else None
        return steps
