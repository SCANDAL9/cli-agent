"""
Unit tests for the Git Worktree Sandboxing and Atomic Diff Engine module.
"""

import subprocess
from pathlib import Path

import pytest

from cli_agent.tools import (
    edit_file,
    get_active_sandbox,
    read_file,
    run_command,
    set_active_sandbox,
    write_file,
)
from cli_agent.worktree import (
    DiffViewer,
    WorktreeManager,
    _get_repo_root,
    _has_head_commit,
    _is_git_repo,
)


@pytest.fixture
def temp_git_repo(tmp_path: Path) -> Path:
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    subprocess.run(["git", "init"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "ci@example.com"], cwd=repo_dir, check=True)
    subprocess.run(["git", "config", "user.name", "CI Bot"], cwd=repo_dir, check=True)

    init_file = repo_dir / "app.py"
    init_file.write_text("def hello() -> str:\n    return 'initial'\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True)
    subprocess.run(["git", "commit", "-m", "initial commit"], cwd=repo_dir, check=True)
    return repo_dir


@pytest.fixture
def empty_git_repo(tmp_path: Path) -> Path:
    repo_dir = tmp_path / "empty_repo"
    repo_dir.mkdir()
    subprocess.run(["git", "init"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "ci@example.com"], cwd=repo_dir, check=True)
    subprocess.run(["git", "config", "user.name", "CI Bot"], cwd=repo_dir, check=True)
    return repo_dir


@pytest.mark.unit
class TestGitWorktreeLifecycle:
    def test_git_repo_detection(self, temp_git_repo: Path, tmp_path: Path) -> None:
        assert _is_git_repo(temp_git_repo) is True
        assert _has_head_commit(temp_git_repo) is True
        assert _get_repo_root(temp_git_repo) == temp_git_repo

        non_repo = tmp_path / "not_git"
        non_repo.mkdir()
        assert _is_git_repo(non_repo) is False
        assert _has_head_commit(non_repo) is False

    def test_empty_git_repo_detection(self, empty_git_repo: Path) -> None:
        assert _is_git_repo(empty_git_repo) is True
        assert _has_head_commit(empty_git_repo) is False

    def test_create_session_and_cleanup(self, temp_git_repo: Path, tmp_path: Path) -> None:
        manager = WorktreeManager(base_dir=tmp_path / "worktrees")
        session = manager.create_session(repo_path=temp_git_repo)

        assert session.is_active is True
        assert session.worktree_dir.exists()
        assert (session.worktree_dir / "app.py").exists()

        # Check git branch was created
        branches = subprocess.run(
            ["git", "branch"], cwd=temp_git_repo, capture_output=True, text=True, check=True
        ).stdout
        assert session.branch_name in branches

        session.cleanup()
        assert session.is_active is False
        assert not session.worktree_dir.exists()

    def test_create_session_on_orphan_repo(self, empty_git_repo: Path, tmp_path: Path) -> None:
        (empty_git_repo / "uncommitted.txt").write_text("draft code", encoding="utf-8")
        manager = WorktreeManager(base_dir=tmp_path / "worktrees_orphan")
        session = manager.create_session(repo_path=empty_git_repo)

        assert session.is_active is True
        assert (session.worktree_dir / "uncommitted.txt").exists()
        session.cleanup()

    def test_create_session_fails_on_non_git_dir(self, tmp_path: Path) -> None:
        non_repo = tmp_path / "plain_dir"
        non_repo.mkdir()
        manager = WorktreeManager()
        with pytest.raises(RuntimeError, match="not inside a git repository"):
            manager.create_session(repo_path=non_repo)

    def test_context_manager_cleanup(self, temp_git_repo: Path, tmp_path: Path) -> None:
        manager = WorktreeManager(base_dir=tmp_path / "worktrees_ctx")
        with manager.create_session(repo_path=temp_git_repo) as session:
            assert session.worktree_dir.exists()
            wt_path = session.worktree_dir

        assert not wt_path.exists()


@pytest.mark.unit
class TestDiffAndPatchApplication:
    def test_diff_generation_and_summary(self, temp_git_repo: Path, tmp_path: Path) -> None:
        manager = WorktreeManager(base_dir=tmp_path / "worktrees_diff")
        session = manager.create_session(repo_path=temp_git_repo)

        # Modify app.py
        app_file = session.worktree_dir / "app.py"
        app_file.write_text("def hello() -> str:\n    return 'sandbox_v2'\n", encoding="utf-8")

        # Create new_mod.py
        new_file = session.worktree_dir / "new_mod.py"
        new_file.write_text("print('hello sandbox')\n", encoding="utf-8")

        summary = session.get_summary()
        assert summary.files_changed == 2
        assert summary.insertions >= 2
        assert "app.py" in summary.files
        assert "new_mod.py" in summary.files

        # Apply to main repo
        apply_res = session.apply_to_main()
        assert apply_res["status"] == "applied"

        # Verify main repo has changes
        main_app = temp_git_repo / "app.py"
        assert "sandbox_v2" in main_app.read_text(encoding="utf-8")
        main_new = temp_git_repo / "new_mod.py"
        assert main_new.exists()

        session.cleanup()

    def test_apply_with_no_changes(self, temp_git_repo: Path, tmp_path: Path) -> None:
        manager = WorktreeManager(base_dir=tmp_path / "worktrees_clean")
        session = manager.create_session(repo_path=temp_git_repo)

        res = session.apply_to_main()
        assert res["status"] == "no_changes"

        session.cleanup()
        assert session.apply_to_main()["status"] == "error"


@pytest.mark.unit
class TestDiffViewer:
    def test_diff_viewer_render_with_and_without_changes(self) -> None:
        empty_panel = DiffViewer.render("")
        assert empty_panel is not None

        diff_text = """diff --git a/foo.py b/foo.py
--- a/foo.py
+++ b/foo.py
@@ -1 +1 @@
-old
+new
"""
        syntax_panel = DiffViewer.render(diff_text)
        assert syntax_panel is not None

    def test_diff_viewer_summary_table(self) -> None:
        from cli_agent.worktree import DiffSummary

        summary = DiffSummary(
            files_changed=2,
            insertions=10,
            deletions=3,
            files=["mod1.py", "mod2.py"],
            diff_text="",
        )
        table = DiffViewer.render_summary(summary)
        assert table is not None
        assert table.row_count == 3


@pytest.mark.unit
class TestWorktreeManagerCleanupAll:
    def test_cleanup_all_removes_multiple_sessions(
        self, temp_git_repo: Path, tmp_path: Path
    ) -> None:
        manager = WorktreeManager(base_dir=tmp_path / "worktrees_multi")
        s1 = manager.create_session(repo_path=temp_git_repo)
        s2 = manager.create_session(repo_path=temp_git_repo)
        assert len(manager._active_sessions) == 2

        manager.cleanup_all()
        assert len(manager._active_sessions) == 0
        assert not s1.worktree_dir.exists()
        assert not s2.worktree_dir.exists()


@pytest.mark.unit
class TestSandboxToolRedirection:
    def test_tools_operate_inside_sandbox_when_active(
        self, temp_git_repo: Path, tmp_path: Path
    ) -> None:
        manager = WorktreeManager(base_dir=tmp_path / "worktrees_tools")
        session = manager.create_session(repo_path=temp_git_repo)

        set_active_sandbox(session.worktree_dir)
        assert get_active_sandbox() == session.worktree_dir

        try:
            # write_file inside sandbox
            w_res = write_file("sandbox_only.py", "CONSTANT = 100\n")
            assert w_res["status"] == "success"
            assert (session.worktree_dir / "sandbox_only.py").exists()
            assert not (temp_git_repo / "sandbox_only.py").exists()

            # edit_file inside sandbox
            edit_res = edit_file(
                "app.py",
                target="return 'initial'",
                replacement="return 'sandboxed'",
            )
            assert edit_res["status"] == "success"
            assert "sandboxed" in (session.worktree_dir / "app.py").read_text(encoding="utf-8")
            assert "initial" in (temp_git_repo / "app.py").read_text(encoding="utf-8")

            # read_file reads from sandbox
            r_res = read_file("app.py")
            assert "sandboxed" in r_res["content"]

            # Absolute path outside cwd is preserved
            outside = tmp_path / "outside.txt"
            outside.write_text("outside", encoding="utf-8")
            assert "outside" in read_file(str(outside))["content"]

            # run_command executes inside sandbox directory by default
            cmd_res = run_command("pwd")
            assert str(session.worktree_dir) in cmd_res.get("stdout", "")

        finally:
            set_active_sandbox(None)
            session.cleanup()

        assert get_active_sandbox() is None
