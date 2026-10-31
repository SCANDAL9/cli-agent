"""
Unit tests for Phase 2 of Wave 5: Per-Worker Worktree Isolation, Hard allowed_files ACL,
and AST-Aware Three-Way Merger (src/cli_agent/tools.py & src/cli_agent/worktree.py).
"""

from __future__ import annotations

import ast
import asyncio
import subprocess
from pathlib import Path

import pytest

from cli_agent.tools import (
    ToolScopeViolationError,
    _enforce_write_scope,
    edit_file,
    get_active_sandbox,
    read_file,
    worker_execution_scope,
    write_file,
)
from cli_agent.worktree import (
    ASTThreeWayMerger,
    WorktreeManager,
)


@pytest.fixture
def temp_git_repo(tmp_path: Path) -> Path:
    repo_dir = tmp_path / "merge_repo"
    repo_dir.mkdir()
    subprocess.run(["git", "init"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "ci@example.com"], cwd=repo_dir, check=True)
    subprocess.run(["git", "config", "user.name", "CI Bot"], cwd=repo_dir, check=True)

    shared_code = (
        "def func_alpha(x: int) -> int:\n"
        "    return x + 1\n\n\n"
        "def func_beta(y: int) -> int:\n"
        "    return y * 2\n"
    )
    (repo_dir / "shared.py").write_text(shared_code, encoding="utf-8")
    (repo_dir / "only_a.py").write_text("VAL_A = 1\n", encoding="utf-8")
    (repo_dir / "only_b.py").write_text("VAL_B = 2\n", encoding="utf-8")

    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True)
    subprocess.run(["git", "commit", "-m", "base commit"], cwd=repo_dir, check=True)
    return repo_dir


@pytest.mark.unit
class TestWorkerExecutionScopeAndACL:
    def test_enforce_write_scope_blocks_unapproved_files_with_zero_bytes_written(
        self, tmp_path: Path
    ) -> None:
        sandbox = tmp_path / "worker_wt"
        sandbox.mkdir()
        approved_file = sandbox / "approved.py"
        forbidden_file = sandbox / "forbidden.py"
        forbidden_file.write_text("ORIGINAL = True\n", encoding="utf-8")

        with worker_execution_scope(
            sandbox_dir=sandbox,
            allowed_files=["approved.py"],
            worker_id="node-worker-1",
        ):
            assert get_active_sandbox() == sandbox.resolve()

            # 1. Approved write succeeds
            ok_res = write_file("approved.py", "APPROVED = 1\n")
            assert ok_res.get("status") == "success"
            assert approved_file.read_text(encoding="utf-8") == "APPROVED = 1\n"

            # 2. Unapproved write_file is rejected with structured ACL violation
            bad_write = write_file("unapproved_new.py", "HACK = 1\n")
            assert bad_write.get("violation_type") == "ALLOWED_FILES_ACL_VIOLATION"
            assert bad_write.get("attempted_path") == "unapproved_new.py"
            assert bad_write.get("allowed_files") == ["approved.py"]
            assert "node-worker-1" in bad_write.get("error", "")
            assert not (sandbox / "unapproved_new.py").exists()

            # 3. Unapproved edit_file is rejected and leaves existing file untouched
            bad_edit = edit_file(
                "forbidden.py", target="ORIGINAL = True", replacement="ORIGINAL = False"
            )
            assert bad_edit.get("violation_type") == "ALLOWED_FILES_ACL_VIOLATION"
            assert forbidden_file.read_text(encoding="utf-8") == "ORIGINAL = True\n"

            # 4. Path traversal outside sandbox is blocked
            escape_res = write_file("../escaped.py", "pwned\n")
            assert escape_res.get("violation_type") == "PATH_TRAVERSAL_VIOLATION"
            assert not (tmp_path / "escaped.py").exists()

        assert get_active_sandbox() is None

    def test_enforce_write_scope_direct_exception(self, tmp_path: Path) -> None:
        sandbox = tmp_path / "direct_wt"
        sandbox.mkdir()
        with worker_execution_scope(
            sandbox_dir=sandbox,
            allowed_files=["src/allowed.py"],
            worker_id="w-direct",
        ):
            with pytest.raises(ToolScopeViolationError) as exc_info:
                _enforce_write_scope("src/blocked.py", sandbox / "src" / "blocked.py")
            assert exc_info.value.violation_type == "ALLOWED_FILES_ACL_VIOLATION"

    @pytest.mark.asyncio
    async def test_concurrent_worker_scopes_have_zero_contextvar_leakage(
        self, tmp_path: Path
    ) -> None:
        wt_1 = tmp_path / "wt_1"
        wt_2 = tmp_path / "wt_2"
        wt_1.mkdir()
        wt_2.mkdir()

        async def run_worker_1() -> None:
            with worker_execution_scope(
                sandbox_dir=wt_1, allowed_files=["mod1.py"], worker_id="w1"
            ):
                await asyncio.sleep(0.02)
                assert get_active_sandbox() == wt_1.resolve()
                r1 = write_file("mod1.py", "W = 1\n")
                assert r1.get("status") == "success"
                r_bad = write_file("mod2.py", "W = 1\n")
                assert r_bad.get("violation_type") == "ALLOWED_FILES_ACL_VIOLATION"

        async def run_worker_2() -> None:
            with worker_execution_scope(
                sandbox_dir=wt_2, allowed_files=["mod2.py"], worker_id="w2"
            ):
                await asyncio.sleep(0.01)
                assert get_active_sandbox() == wt_2.resolve()
                r2 = write_file("mod2.py", "W = 2\n")
                assert r2.get("status") == "success"
                r_bad = write_file("mod1.py", "W = 2\n")
                assert r_bad.get("violation_type") == "ALLOWED_FILES_ACL_VIOLATION"
                await asyncio.sleep(0.02)
                assert read_file("mod2.py")["content"].endswith("W = 2\n")

        await asyncio.gather(run_worker_1(), run_worker_2())
        assert (wt_1 / "mod1.py").exists()
        assert not (wt_1 / "mod2.py").exists()
        assert (wt_2 / "mod2.py").exists()
        assert not (wt_2 / "mod1.py").exists()


@pytest.mark.unit
class TestASTThreeWayMerger:
    def test_create_worker_session_and_upstream_premerge(
        self, temp_git_repo: Path, tmp_path: Path
    ) -> None:
        wm = WorktreeManager(base_dir=tmp_path / "wt_sessions")
        try:
            s_up = wm.create_worker_session(
                plan_id="p100", node_id="node-up", repo_path=temp_git_repo
            )
            assert s_up.branch_name == "swarm/p100/node-up"
            (s_up.worktree_dir / "upstream_new.py").write_text("UP = 42\n", encoding="utf-8")

            s_down = wm.create_worker_session(
                plan_id="p100",
                node_id="node-down",
                repo_path=temp_git_repo,
                upstream_sessions=[s_up],
            )
            assert (s_down.worktree_dir / "upstream_new.py").exists()
            assert (s_down.worktree_dir / "upstream_new.py").read_text(
                encoding="utf-8"
            ) == "UP = 42\n"
        finally:
            wm.cleanup_all()

    def test_stage1_disjoint_files_fast_forward(self, temp_git_repo: Path, tmp_path: Path) -> None:
        wm = WorktreeManager(base_dir=tmp_path / "wt_disjoint")
        try:
            base_s = wm.create_worker_session("p1", "base", repo_path=temp_git_repo)
            w1 = wm.create_worker_session("p1", "w1", repo_path=temp_git_repo)
            w2 = wm.create_worker_session("p1", "w2", repo_path=temp_git_repo)
            int_s = wm.create_worker_session("p1", "int", repo_path=temp_git_repo)

            (w1.worktree_dir / "only_a.py").write_text("VAL_A = 100\n", encoding="utf-8")
            (w1.worktree_dir / "created_by_w1.py").write_text("NEW = True\n", encoding="utf-8")
            (w2.worktree_dir / "only_b.py").unlink()

            merger = ASTThreeWayMerger(temp_git_repo)
            res = merger.merge_worker_sessions(
                base_session=base_s,
                worker_sessions={"w1": w1, "w2": w2},
                integration_session=int_s,
            )

            assert res.status == "clean"
            assert sorted(res.merged_files) == ["created_by_w1.py", "only_a.py", "only_b.py"]
            assert (int_s.worktree_dir / "only_a.py").read_text(encoding="utf-8") == "VAL_A = 100\n"
            assert (int_s.worktree_dir / "created_by_w1.py").exists()
            assert not (int_s.worktree_dir / "only_b.py").exists()
        finally:
            wm.cleanup_all()

    def test_stage2_disjoint_ast_symbols_in_shared_file_with_eof_append_splicing(
        self, temp_git_repo: Path, tmp_path: Path
    ) -> None:
        wm = WorktreeManager(base_dir=tmp_path / "wt_ast_splice")
        try:
            base_s = wm.create_worker_session("p2", "base", repo_path=temp_git_repo)
            w1 = wm.create_worker_session("p2", "w1", repo_path=temp_git_repo)
            w2 = wm.create_worker_session("p2", "w2", repo_path=temp_git_repo)
            int_s = wm.create_worker_session("p2", "int", repo_path=temp_git_repo)

            # Worker 1 modifies func_alpha AND appends helper_w1 at EOF
            w1_code = (
                "import os\n\n"
                "def func_alpha(x: int) -> int:\n"
                "    return x + 100\n\n\n"
                "def func_beta(y: int) -> int:\n"
                "    return y * 2\n\n\n"
                "def helper_w1() -> str:\n"
                "    return os.name\n"
            )
            # Worker 2 modifies func_beta AND appends helper_w2 at the exact same EOF position
            w2_code = (
                "import sys\n\n"
                "def func_alpha(x: int) -> int:\n"
                "    return x + 1\n\n\n"
                "def func_beta(y: int) -> int:\n"
                "    return y * 200\n\n\n"
                "def helper_w2() -> str:\n"
                "    return sys.platform\n"
            )
            (w1.worktree_dir / "shared.py").write_text(w1_code, encoding="utf-8")
            (w2.worktree_dir / "shared.py").write_text(w2_code, encoding="utf-8")

            merger = ASTThreeWayMerger(temp_git_repo)
            res = merger.merge_worker_sessions(
                base_session=base_s,
                worker_sessions={"w1": w1, "w2": w2},
                integration_session=int_s,
            )

            assert res.status == "ast_resolved"
            assert "shared.py" in res.ast_resolved_files
            assert not res.conflicts

            merged_source = (int_s.worktree_dir / "shared.py").read_text(encoding="utf-8")
            # Must parse cleanly as valid Python AST
            ast.parse(merged_source)
            assert "return x + 100" in merged_source
            assert "return y * 200" in merged_source
            assert "def helper_w1()" in merged_source
            assert "def helper_w2()" in merged_source
            assert "import os" in merged_source
            assert "import sys" in merged_source
        finally:
            wm.cleanup_all()

    def test_stage2_overlapping_ast_symbol_mutation_flags_conflict(
        self, temp_git_repo: Path, tmp_path: Path
    ) -> None:
        wm = WorktreeManager(base_dir=tmp_path / "wt_ast_conflict")
        try:
            base_s = wm.create_worker_session("p3", "base", repo_path=temp_git_repo)
            w1 = wm.create_worker_session("p3", "w1", repo_path=temp_git_repo)
            w2 = wm.create_worker_session("p3", "w2", repo_path=temp_git_repo)
            int_s = wm.create_worker_session("p3", "int", repo_path=temp_git_repo)

            # Both workers mutate func_alpha differently
            (w1.worktree_dir / "shared.py").write_text(
                "def func_alpha(x: int) -> int:\n    return x + 10\n\n\n"
                "def func_beta(y: int) -> int:\n    return y * 2\n",
                encoding="utf-8",
            )
            (w2.worktree_dir / "shared.py").write_text(
                "def func_alpha(x: int) -> int:\n    return x + 99\n\n\n"
                "def func_beta(y: int) -> int:\n    return y * 2\n",
                encoding="utf-8",
            )

            merger = ASTThreeWayMerger(temp_git_repo)
            res = merger.merge_worker_sessions(
                base_session=base_s,
                worker_sessions={"w1": w1, "w2": w2},
                integration_session=int_s,
            )

            assert res.status == "conflict"
            assert len(res.conflicts) == 1
            conflict = res.conflicts[0]
            assert conflict.file_path == "shared.py"
            assert conflict.colliding_symbols == ["func_alpha"]
            assert conflict.worker_a_id == "w1"
            assert conflict.worker_b_id == "w2"
            assert "return x + 1" in conflict.base_source
            assert "return x + 10" in conflict.worker_a_source
            assert "return x + 99" in conflict.worker_b_source
        finally:
            wm.cleanup_all()
