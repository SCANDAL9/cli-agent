"""Tests for local authenticated worker leases and diff application."""

import asyncio
import hashlib
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from cli_agent.security import RiskLevel, SecurityPolicy
from cli_agent.storage.turso import TursoLocalEngine
from cli_agent.team import MultiAgentTeam
from cli_agent.worker_protocol import (
    DistributedWorkerCoordinator,
    WorkerLeaseError,
    WorkerLeaseUnavailable,
)

PATCH = """diff --git a/result.txt b/result.txt
--- a/result.txt
+++ b/result.txt
@@ -1 +1 @@
-before
+after
"""


class Clock:
    def __init__(self, value: float = 1_000.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def _make_repo(path):
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    (path / "result.txt").write_text("before\n", encoding="utf-8")
    subprocess.run(["git", "add", "result.txt"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "baseline"], cwd=path, check=True)


def _coordinator(tmp_path, repo, policy=None, clock=None, database="leases.db"):
    return DistributedWorkerCoordinator(
        storage=TursoLocalEngine(tmp_path / database),
        security_policy=policy or SecurityPolicy(),
        repo_root=repo,
        clock=clock,
    )


def test_claim_persists_generation_and_rejects_a_second_active_claim(tmp_path):
    repo = tmp_path / "repo"
    _make_repo(repo)
    clock = Clock()
    coordinator = _coordinator(tmp_path, repo, clock=clock)
    worker = coordinator.register_worker("worker-a")

    lease = coordinator.claim_task("task-a", worker, lease_seconds=30)

    assert lease.generation == 1
    row = coordinator.storage.execute(
        "SELECT task_id, worker_id, generation, expires_at, status FROM worker_task_leases"
    )[0]
    assert row == {
        "task_id": "task-a",
        "worker_id": "worker-a",
        "generation": 1,
        "expires_at": 1_030.0,
        "status": "claimed",
    }
    with pytest.raises(WorkerLeaseUnavailable):
        coordinator.claim_task("task-a", worker, lease_seconds=30)


def test_team_exposes_the_local_virtual_worker_protocol(tmp_path):
    repo = tmp_path / "repo"
    _make_repo(repo)
    coordinator = _coordinator(tmp_path, repo)
    team = MultiAgentTeam(client=object(), worker_coordinator=coordinator)
    worker = team.register_virtual_worker("worker-a")

    lease = team.claim_worker_task("task-a", worker)

    assert lease.worker_id == worker.worker_id
    assert lease.generation == 1
    with pytest.raises(RuntimeError, match="not configured"):
        MultiAgentTeam(client=object()).register_virtual_worker()


def test_concurrent_local_workers_cannot_claim_the_same_task(tmp_path):
    repo = tmp_path / "repo"
    _make_repo(repo)
    coordinator = _coordinator(tmp_path, repo, database="shared-leases.db")
    workers = [coordinator.register_worker(worker_id) for worker_id in ("worker-a", "worker-b")]

    def claim(worker):
        try:
            return coordinator.claim_task("exclusive-task", worker, lease_seconds=30)
        except WorkerLeaseUnavailable:
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(claim, workers))

    assert sum(result is not None for result in results) == 1


def test_expired_lease_reclaims_with_new_generation_and_rejects_stale_result(tmp_path):
    repo = tmp_path / "repo"
    _make_repo(repo)
    clock = Clock()
    coordinator = _coordinator(tmp_path, repo, clock=clock)
    first_worker = coordinator.register_worker("worker-a")
    second_worker = coordinator.register_worker("worker-b")
    old_lease = coordinator.claim_task("task-a", first_worker, lease_seconds=5)
    clock.value = 1_006.0

    new_lease = coordinator.claim_task("task-a", second_worker, lease_seconds=30)

    assert new_lease.generation == old_lease.generation + 1
    with pytest.raises(WorkerLeaseError):
        asyncio.run(
            coordinator.apply_worker_diff(
                old_lease,
                first_worker,
                PATCH,
                first_worker.sign_result(old_lease, PATCH),
            )
        )
    assert (repo / "result.txt").read_text(encoding="utf-8") == "before\n"


@pytest.mark.asyncio
async def test_forged_result_signature_is_rejected_before_policy_or_workspace_mutation(tmp_path):
    repo = tmp_path / "repo"
    _make_repo(repo)
    observed = []
    policy = SecurityPolicy(confirm_callback=lambda *args: observed.append(args) or True)
    coordinator = _coordinator(tmp_path, repo, policy=policy)
    worker = coordinator.register_worker("worker-a")
    lease = coordinator.claim_task("task-a", worker)

    with pytest.raises(WorkerLeaseError):
        await coordinator.apply_worker_diff(
            lease,
            worker,
            PATCH + "# altered after signing\n",
            worker.sign_result(lease, PATCH),
        )

    assert not observed
    assert (repo / "result.txt").read_text(encoding="utf-8") == "before\n"


@pytest.mark.asyncio
async def test_tampered_lease_and_unregistered_worker_are_rejected(tmp_path):
    repo = tmp_path / "repo"
    _make_repo(repo)
    coordinator = _coordinator(tmp_path, repo)
    worker = coordinator.register_worker("worker-a")
    other_worker = coordinator.register_worker("worker-b")
    lease = coordinator.claim_task("task-a", worker)

    with pytest.raises(WorkerLeaseError):
        await coordinator.apply_worker_diff(
            replace(lease, generation=lease.generation + 1),
            worker,
            PATCH,
            worker.sign_result(lease, PATCH),
        )
    with pytest.raises(WorkerLeaseError):
        await coordinator.apply_worker_diff(lease, other_worker, PATCH, "invalid-signature")

    assert (repo / "result.txt").read_text(encoding="utf-8") == "before\n"


@pytest.mark.asyncio
async def test_confirmation_callback_cannot_change_the_approved_patch(tmp_path):
    repo = tmp_path / "repo"
    _make_repo(repo)

    def mutate_approved_diff(_name, arguments, _risk):
        arguments["diff"] = "different patch"
        return True

    coordinator = _coordinator(
        tmp_path,
        repo,
        policy=SecurityPolicy(confirm_callback=mutate_approved_diff),
    )
    worker = coordinator.register_worker("worker-a")
    lease = coordinator.claim_task("task-a", worker)
    result = await coordinator.apply_worker_diff(
        lease,
        worker,
        PATCH,
        worker.sign_result(lease, PATCH),
    )

    assert result.status == "denied"
    assert (repo / "result.txt").read_text(encoding="utf-8") == "before\n"


@pytest.mark.asyncio
async def test_exact_diff_requires_policy_approval_and_applies_exact_signed_bytes(tmp_path):
    repo = tmp_path / "repo"
    _make_repo(repo)
    observed = []
    allow = False

    def confirm(name, arguments, risk):
        observed.append((name, arguments.copy(), risk))
        return allow

    policy = SecurityPolicy(auto_approve=True, confirm_callback=confirm)
    coordinator = _coordinator(tmp_path, repo, policy=policy)
    worker = coordinator.register_worker("worker-a")
    lease = coordinator.claim_task("task-a", worker)
    signature = worker.sign_result(lease, PATCH)
    initial_hash = hashlib.sha256((repo / "result.txt").read_bytes()).hexdigest()

    denied = await coordinator.apply_worker_diff(lease, worker, PATCH, signature)

    assert denied.status == "denied"
    assert hashlib.sha256((repo / "result.txt").read_bytes()).hexdigest() == initial_hash
    assert observed[-1][0] == "apply_worker_diff"
    assert observed[-1][1]["diff"] == PATCH
    assert observed[-1][1]["diff_sha256"] == hashlib.sha256(PATCH.encode()).hexdigest()
    assert observed[-1][2] == RiskLevel.HIGH

    allow = True
    applied = await coordinator.apply_worker_diff(lease, worker, PATCH, signature)

    assert applied.status == "applied"
    assert (repo / "result.txt").read_text(encoding="utf-8") == "after\n"
    assert coordinator.storage.execute(
        "SELECT status, result_hash FROM worker_task_leases WHERE task_id = ?", ("task-a",)
    )[0] == {
        "status": "applied",
        "result_hash": hashlib.sha256(PATCH.encode()).hexdigest(),
    }


@pytest.mark.asyncio
async def test_lease_expiry_during_approval_prevents_application(tmp_path):
    repo = tmp_path / "repo"
    _make_repo(repo)
    clock = Clock()

    def approve_after_expiry(_name, _arguments, _risk):
        clock.value = 1_006.0
        return True

    coordinator = _coordinator(
        tmp_path,
        repo,
        policy=SecurityPolicy(confirm_callback=approve_after_expiry),
        clock=clock,
    )
    worker = coordinator.register_worker("worker-a")
    lease = coordinator.claim_task("task-a", worker, lease_seconds=5)

    with pytest.raises(WorkerLeaseError):
        await coordinator.apply_worker_diff(
            lease,
            worker,
            PATCH,
            worker.sign_result(lease, PATCH),
        )

    assert (repo / "result.txt").read_text(encoding="utf-8") == "before\n"
