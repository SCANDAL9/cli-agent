"""Local authenticated leases for distributed-worker protocol boundaries.

This module deliberately implements no network transport or durable trust root.
Worker credentials and the coordinator signing key exist only for one process
lifetime; lease generations and state are persisted in a local Turso database.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import secrets
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from cli_agent.security import SecurityPolicy
from cli_agent.storage.base import StorageEngine


class WorkerLeaseError(RuntimeError):
    """A worker identity, lease, or signed result failed protocol validation."""


class WorkerLeaseUnavailable(WorkerLeaseError):
    """The requested task currently has a live claim or has already completed."""


@dataclass(frozen=True)
class WorkerLease:
    task_id: str
    worker_id: str
    lease_id: str
    generation: int
    expires_at: float
    signature: str


@dataclass(frozen=True)
class WorkerCredential:
    """Ephemeral local worker identity; the secret is never persisted."""

    worker_id: str
    _secret: bytes = field(repr=False, compare=False)

    def sign_result(self, lease: WorkerLease, diff: str) -> str:
        if lease.worker_id != self.worker_id:
            raise WorkerLeaseError("Worker credential does not own this lease.")
        diff_hash = hashlib.sha256(diff.encode("utf-8")).hexdigest()
        payload = _result_payload(lease, diff_hash)
        return hmac.new(self._secret, payload, hashlib.sha256).hexdigest()


@dataclass(frozen=True)
class WorkerApplyResult:
    status: Literal["applied", "denied", "conflict", "error"]
    diff_sha256: str
    message: str = ""


class DistributedWorkerCoordinator:
    """Coordinates local virtual workers without opening a remote transport."""

    def __init__(
        self,
        storage: StorageEngine,
        security_policy: SecurityPolicy,
        repo_root: Path,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.storage = storage
        self.security_policy = security_policy
        self.repo_root = Path(repo_root).expanduser().resolve()
        self._clock = clock or time.time
        self._lease_signing_key = secrets.token_bytes(32)
        self._worker_secrets: dict[str, bytes] = {}
        self.storage.execute(
            """
            CREATE TABLE IF NOT EXISTS worker_task_leases (
                task_id TEXT PRIMARY KEY,
                worker_id TEXT NOT NULL,
                lease_id TEXT NOT NULL,
                generation INTEGER NOT NULL,
                claimed_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('claimed', 'applying', 'applied', 'failed')),
                result_hash TEXT
            )
            """
        )

    def register_worker(self, worker_id: str | None = None) -> WorkerCredential:
        """Registers an in-process virtual worker with a fresh unpersisted secret."""
        resolved_id = worker_id or f"local-{uuid4().hex}"
        if not resolved_id.strip():
            raise ValueError("Worker ID cannot be empty.")
        if resolved_id in self._worker_secrets:
            raise ValueError(f"Worker ID '{resolved_id}' is already registered.")
        secret = secrets.token_bytes(32)
        self._worker_secrets[resolved_id] = secret
        return WorkerCredential(resolved_id, secret)

    def claim_task(
        self,
        task_id: str,
        worker: WorkerCredential,
        *,
        lease_seconds: float = 60.0,
    ) -> WorkerLease:
        """Atomically claims a task, incrementing its generation on reclaim."""
        if not task_id.strip():
            raise ValueError("Task ID cannot be empty.")
        if (
            isinstance(lease_seconds, bool)
            or not isinstance(lease_seconds, (int, float))
            or not math.isfinite(lease_seconds)
            or lease_seconds <= 0
        ):
            raise ValueError("Lease duration must be a finite positive number of seconds.")
        self._authenticate_worker(worker)
        now = float(self._clock())
        if not math.isfinite(now):
            raise ValueError("Lease clock must return a finite timestamp.")
        lease_id = uuid4().hex
        expires_at = now + float(lease_seconds)
        rows = self.storage.execute(
            """
            INSERT INTO worker_task_leases (
                task_id, worker_id, lease_id, generation, claimed_at,
                expires_at, status, result_hash
            ) VALUES (?, ?, ?, 1, ?, ?, 'claimed', NULL)
            ON CONFLICT(task_id) DO UPDATE SET
                worker_id = excluded.worker_id,
                lease_id = excluded.lease_id,
                generation = worker_task_leases.generation + 1,
                claimed_at = excluded.claimed_at,
                expires_at = excluded.expires_at,
                status = 'claimed',
                result_hash = NULL
            WHERE worker_task_leases.status = 'failed'
               OR (worker_task_leases.status = 'claimed'
                   AND worker_task_leases.expires_at <= excluded.claimed_at)
            RETURNING generation
            """,
            (task_id, worker.worker_id, lease_id, now, expires_at),
        )
        if not rows:
            raise WorkerLeaseUnavailable(
                f"Task '{task_id}' already has a live, applying, or completed lease."
            )
        generation = int(rows[0]["generation"])
        unsigned = WorkerLease(
            task_id=task_id,
            worker_id=worker.worker_id,
            lease_id=lease_id,
            generation=generation,
            expires_at=expires_at,
            signature="",
        )
        signature = hmac.new(
            self._lease_signing_key, _lease_payload(unsigned), hashlib.sha256
        ).hexdigest()
        return WorkerLease(**{**unsigned.__dict__, "signature": signature})

    async def apply_worker_diff(
        self,
        lease: WorkerLease,
        worker: WorkerCredential,
        diff: str,
        result_signature: str,
    ) -> WorkerApplyResult:
        """Approves and applies only the exact authenticated diff bytes."""
        if not isinstance(diff, str) or not diff.strip():
            raise WorkerLeaseError("Worker result must contain a non-empty UTF-8 diff.")
        diff_bytes = diff.encode("utf-8")
        diff_hash = hashlib.sha256(diff_bytes).hexdigest()
        worker_secret = self._authenticate_worker(worker)
        self._validate_lease(lease, worker)
        expected_result_signature = hmac.new(
            worker_secret, _result_payload(lease, diff_hash), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(expected_result_signature, result_signature):
            raise WorkerLeaseError("Worker result signature does not match the submitted diff.")

        permission_arguments: dict[str, Any] = {
            "task_id": lease.task_id,
            "worker_id": lease.worker_id,
            "lease_id": lease.lease_id,
            "generation": lease.generation,
            "expires_at": lease.expires_at,
            "diff_sha256": diff_hash,
            "diff": diff,
        }
        approved, reason = await self.security_policy.check_permission_async(
            "apply_worker_diff", permission_arguments
        )
        if permission_arguments != {
            "task_id": lease.task_id,
            "worker_id": lease.worker_id,
            "lease_id": lease.lease_id,
            "generation": lease.generation,
            "expires_at": lease.expires_at,
            "diff_sha256": diff_hash,
            "diff": diff,
        }:
            return WorkerApplyResult(
                "denied", diff_hash, "Approval details changed during confirmation."
            )
        if not approved:
            return WorkerApplyResult("denied", diff_hash, reason)

        # Approval can await a human; reject an expired or superseded lease before writes.
        self._validate_lease(lease, worker)
        applying = self.storage.execute(
            """
            UPDATE worker_task_leases SET status = 'applying'
            WHERE task_id = ? AND worker_id = ? AND lease_id = ? AND generation = ?
              AND status = 'claimed' AND expires_at > ?
            RETURNING task_id
            """,
            (
                lease.task_id,
                lease.worker_id,
                lease.lease_id,
                lease.generation,
                float(self._clock()),
            ),
        )
        if not applying:
            raise WorkerLeaseError("Worker lease expired or changed before diff application.")

        try:
            check = subprocess.run(
                ["git", "apply", "--check"],
                input=diff_bytes,
                cwd=self.repo_root,
                capture_output=True,
                check=False,
            )
            if check.returncode:
                self._set_failed(lease)
                message = check.stderr.decode("utf-8", errors="replace").strip()
                return WorkerApplyResult("conflict", diff_hash, message)
            applied = subprocess.run(
                ["git", "apply"],
                input=diff_bytes,
                cwd=self.repo_root,
                capture_output=True,
                check=False,
            )
            if applied.returncode:
                self._set_failed(lease)
                message = applied.stderr.decode("utf-8", errors="replace").strip()
                return WorkerApplyResult("error", diff_hash, message)
        except OSError as exc:
            self._set_failed(lease)
            return WorkerApplyResult("error", diff_hash, str(exc))

        finalized = self.storage.execute(
            """
            UPDATE worker_task_leases SET status = 'applied', result_hash = ?
            WHERE task_id = ? AND lease_id = ? AND generation = ? AND status = 'applying'
            RETURNING task_id
            """,
            (diff_hash, lease.task_id, lease.lease_id, lease.generation),
        )
        if not finalized:
            raise WorkerLeaseError("Diff was applied but its lease record could not be finalized.")
        return WorkerApplyResult("applied", diff_hash)

    def _authenticate_worker(self, worker: WorkerCredential) -> bytes:
        if not isinstance(worker, WorkerCredential):
            raise WorkerLeaseError("A registered worker credential is required.")
        registered = self._worker_secrets.get(worker.worker_id)
        if registered is None or not hmac.compare_digest(registered, worker._secret):
            raise WorkerLeaseError("Worker credential is unknown or no longer valid.")
        return registered

    def _validate_lease(self, lease: WorkerLease, worker: WorkerCredential) -> None:
        secret = self._authenticate_worker(worker)
        if lease.worker_id != worker.worker_id:
            raise WorkerLeaseError("Worker credential does not own this lease.")
        expected_signature = hmac.new(
            self._lease_signing_key, _lease_payload(lease), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(expected_signature, lease.signature):
            raise WorkerLeaseError("Lease signature is invalid.")
        if float(self._clock()) >= lease.expires_at:
            raise WorkerLeaseError("Worker lease has expired.")
        rows = self.storage.execute(
            """
            SELECT worker_id, lease_id, generation, expires_at, status
            FROM worker_task_leases WHERE task_id = ?
            """,
            (lease.task_id,),
        )
        if not rows:
            raise WorkerLeaseError("Worker lease is no longer present in local Turso storage.")
        current = rows[0]
        if (
            current["worker_id"] != lease.worker_id
            or current["lease_id"] != lease.lease_id
            or int(current["generation"]) != lease.generation
            or float(current["expires_at"]) != lease.expires_at
            or current["status"] != "claimed"
        ):
            raise WorkerLeaseError("Worker lease is stale or no longer claimable.")
        # Retain the explicit credential check adjacent to the persisted lease check.
        if not hmac.compare_digest(secret, worker._secret):
            raise WorkerLeaseError("Worker credential changed during lease validation.")

    def _set_failed(self, lease: WorkerLease) -> None:
        self.storage.execute(
            """
            UPDATE worker_task_leases SET status = 'failed'
            WHERE task_id = ? AND lease_id = ? AND generation = ? AND status = 'applying'
            """,
            (lease.task_id, lease.lease_id, lease.generation),
        )


def _lease_payload(lease: WorkerLease) -> bytes:
    return json.dumps(
        {
            "task_id": lease.task_id,
            "worker_id": lease.worker_id,
            "lease_id": lease.lease_id,
            "generation": lease.generation,
            "expires_at": lease.expires_at,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _result_payload(lease: WorkerLease, diff_hash: str) -> bytes:
    return json.dumps(
        {
            "lease": _lease_payload(lease).decode("utf-8"),
            "lease_signature": lease.signature,
            "diff_sha256": diff_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
