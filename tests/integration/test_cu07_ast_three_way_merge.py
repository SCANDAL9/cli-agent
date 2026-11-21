"""
Integration test for CU-07: Lead Architect AST-Aware Three-Way Merge & Semantic Conflict Detection
(Wave 5 Milestone W5.1 & W5.2).
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

from cli_agent.worktree import ASTThreeWayMerger, WorktreeManager


@pytest.fixture
def repo_cu07(tmp_path: Path) -> Path:
    repo = tmp_path / "repo_cu07"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "merge@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Merge Bot"], cwd=repo, check=True)

    base_service = (
        "def compute_alpha(x: int) -> int:\n"
        "    return x + 1\n\n\n"
        "def compute_beta(y: int) -> int:\n"
        "    return y * 2\n"
    )
    (repo / "service.py").write_text(base_service, encoding="utf-8")

    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "initial base commit"], cwd=repo, check=True)
    return repo


@pytest.mark.integration
class TestCU07ASTThreeWayMerge:
    def test_scenario_a_disjoint_ast_symbols_merge_cleanly_into_integration_session(
        self, repo_cu07: Path, tmp_path: Path
    ) -> None:
        wm = WorktreeManager(base_dir=tmp_path / "wt_cu07_a")
        try:
            base_s = wm.create_worker_session("p7a", "base", repo_path=repo_cu07)
            w1 = wm.create_worker_session("p7a", "w1", repo_path=repo_cu07)
            w2 = wm.create_worker_session("p7a", "w2", repo_path=repo_cu07)
            int_s = wm.create_worker_session("p7a", "int", repo_path=repo_cu07)

            # Worker 1: modifies compute_alpha
            (w1.worktree_dir / "service.py").write_text(
                "def compute_alpha(x: int) -> int:\n"
                "    return x + 1000\n\n\n"
                "def compute_beta(y: int) -> int:\n"
                "    return y * 2\n",
                encoding="utf-8",
            )

            # Worker 2: modifies compute_beta and adds compute_gamma at EOF
            (w2.worktree_dir / "service.py").write_text(
                "def compute_alpha(x: int) -> int:\n"
                "    return x + 1\n\n\n"
                "def compute_beta(y: int) -> int:\n"
                "    return y * 999\n\n\n"
                "def compute_gamma(z: int) -> int:\n"
                "    return z ** 3\n",
                encoding="utf-8",
            )

            merger = ASTThreeWayMerger(repo_cu07)
            res = merger.merge_worker_sessions(
                base_session=base_s,
                worker_sessions={"w1": w1, "w2": w2},
                integration_session=int_s,
            )

            assert res.status in ("clean", "ast_resolved")
            assert not res.conflicts

            merged_text = (int_s.worktree_dir / "service.py").read_text(encoding="utf-8")
            parsed = ast.parse(merged_text)
            assert parsed is not None
            assert "return x + 1000" in merged_text
            assert "return y * 999" in merged_text
            assert "def compute_gamma(z: int)" in merged_text
        finally:
            wm.cleanup_all()

    def test_scenario_b_overlapping_symbol_mutation_rejects_with_symbol_conflict(
        self, repo_cu07: Path, tmp_path: Path
    ) -> None:
        wm = WorktreeManager(base_dir=tmp_path / "wt_cu07_b")
        try:
            base_s = wm.create_worker_session("p7b", "base", repo_path=repo_cu07)
            w1 = wm.create_worker_session("p7b", "w1", repo_path=repo_cu07)
            w2 = wm.create_worker_session("p7b", "w2", repo_path=repo_cu07)
            int_s = wm.create_worker_session("p7b", "int", repo_path=repo_cu07)

            # Both workers mutate compute_alpha
            (w1.worktree_dir / "service.py").write_text(
                "def compute_alpha(x: int) -> int:\n"
                "    return x + 10\n\n\n"
                "def compute_beta(y: int) -> int:\n"
                "    return y * 2\n",
                encoding="utf-8",
            )
            (w2.worktree_dir / "service.py").write_text(
                "def compute_alpha(x: int) -> int:\n"
                "    return x + 50\n\n\n"
                "def compute_beta(y: int) -> int:\n"
                "    return y * 2\n",
                encoding="utf-8",
            )

            merger = ASTThreeWayMerger(repo_cu07)
            res = merger.merge_worker_sessions(
                base_session=base_s,
                worker_sessions={"w1": w1, "w2": w2},
                integration_session=int_s,
            )

            assert res.status == "conflict"
            assert len(res.conflicts) == 1
            conflict = res.conflicts[0]
            assert conflict.file_path == "service.py"
            assert conflict.colliding_symbols == ["compute_alpha"]
            assert "return x + 10" in conflict.worker_a_source
            assert "return x + 50" in conflict.worker_b_source
        finally:
            wm.cleanup_all()
