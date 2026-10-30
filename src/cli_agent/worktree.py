"""
Git Worktree Sandboxing and Atomic Diff Engine for cli-agent.
Provides isolated ephemeral branches and worktrees to safeguard the user's active workspace,
with GitHub-style unified diff reviews and atomic patch application.
"""

import os
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console, RenderableType
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

DEFAULT_WORKTREE_BASE = Path("/tmp/cli-agent/worktrees")


def _is_git_repo(path: Path) -> bool:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=path,
            capture_output=True,
            text=True,
        )
        return res.returncode == 0 and res.stdout.strip() == "true"
    except Exception:
        return False


def _has_head_commit(path: Path) -> bool:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=path,
            capture_output=True,
            text=True,
        )
        return res.returncode == 0
    except Exception:
        return False


def _get_repo_root(path: Path) -> Path:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=path,
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(res.stdout.strip()).resolve()
    except Exception:
        return path.resolve()


@dataclass
class DiffSummary:
    files_changed: int
    insertions: int
    deletions: int
    files: list[str]
    diff_text: str


class WorktreeSession:
    """Represents an isolated, ephemeral Git worktree session."""

    def __init__(
        self,
        repo_root: Path,
        worktree_dir: Path,
        branch_name: str,
        has_head: bool = True,
    ):
        self.repo_root = repo_root.resolve()
        self.worktree_dir = worktree_dir.resolve()
        self.branch_name = branch_name
        self.has_head = has_head
        self.is_active = True
        self.created_at = time.time()

    def get_diff(self) -> str:
        """Returns the unified diff representing changes made inside the sandbox."""
        if not self.is_active or not self.worktree_dir.exists():
            return ""

        try:
            # Stage intent-to-add so newly created untracked files appear in git diff
            subprocess.run(
                ["git", "add", "-N", "."],
                cwd=self.worktree_dir,
                capture_output=True,
                check=False,
            )

            cmd = ["git", "diff", "HEAD"] if self.has_head else ["git", "diff"]
            res = subprocess.run(
                cmd,
                cwd=self.worktree_dir,
                capture_output=True,
                text=True,
                check=False,
            )
            return res.stdout
        except Exception:
            return ""

    def get_summary(self) -> DiffSummary:
        """Parses statistics and affected files from the unified diff."""
        diff_text = self.get_diff()
        if not diff_text.strip():
            return DiffSummary(
                files_changed=0,
                insertions=0,
                deletions=0,
                files=[],
                diff_text="",
            )

        files: list[str] = []
        insertions = 0
        deletions = 0

        for line in diff_text.splitlines():
            if line.startswith("diff --git a/"):
                parts = line.split(" ")
                if len(parts) >= 4:
                    file_name = parts[2][2:]  # strip 'a/'
                    files.append(file_name)
            elif line.startswith("+") and not line.startswith("+++"):
                insertions += 1
            elif line.startswith("-") and not line.startswith("---"):
                deletions += 1

        return DiffSummary(
            files_changed=len(files),
            insertions=insertions,
            deletions=deletions,
            files=files,
            diff_text=diff_text,
        )

    def apply_to_main(self) -> dict[str, Any]:
        """Applies the sandbox modifications back into the main workspace atomically."""
        if not self.is_active:
            return {"status": "error", "message": "Session is inactive or already closed."}

        diff_text = self.get_diff()
        if not diff_text.strip():
            return {"status": "no_changes", "message": "No changes found in sandbox to apply."}

        # Check if patch applies cleanly
        check_proc = subprocess.run(
            ["git", "apply", "--check"],
            input=diff_text,
            text=True,
            cwd=self.repo_root,
            capture_output=True,
        )
        if check_proc.returncode != 0:
            return {
                "status": "conflict",
                "message": f"Patch cannot be applied cleanly to main workspace: {check_proc.stderr}",
            }

        apply_proc = subprocess.run(
            ["git", "apply"],
            input=diff_text,
            text=True,
            cwd=self.repo_root,
            capture_output=True,
        )
        if apply_proc.returncode != 0:
            return {
                "status": "error",
                "message": f"Failed applying patch to main workspace: {apply_proc.stderr}",
            }

        summary = self.get_summary()
        return {
            "status": "applied",
            "files_changed": summary.files_changed,
            "insertions": summary.insertions,
            "deletions": summary.deletions,
            "files": summary.files,
        }

    def cleanup(self) -> None:
        """Removes the ephemeral git worktree and deletes the temporary branch."""
        if not self.is_active:
            return

        # Remove git worktree
        try:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(self.worktree_dir)],
                cwd=self.repo_root,
                capture_output=True,
                check=False,
            )
        except Exception:
            pass

        # Delete ephemeral branch
        try:
            subprocess.run(
                ["git", "branch", "-D", self.branch_name],
                cwd=self.repo_root,
                capture_output=True,
                check=False,
            )
        except Exception:
            pass

        # Fallback directory removal if git worktree remove left artifacts
        if self.worktree_dir.exists():
            shutil.rmtree(self.worktree_dir, ignore_errors=True)

        self.is_active = False

    def __enter__(self) -> "WorktreeSession":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        self.cleanup()


class WorktreeManager:
    """Factory and manager for creating and orchestrating ephemeral Git worktrees."""

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or DEFAULT_WORKTREE_BASE
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._active_sessions: list[WorktreeSession] = []

    def create_session(
        self,
        repo_path: Path | str | None = None,
        branch_prefix: str = "agent",
    ) -> WorktreeSession:
        """Creates an isolated git worktree for safe code mutations and tests."""
        target = Path(repo_path or Path.cwd()).resolve()
        if not _is_git_repo(target):
            raise RuntimeError(f"Path '{target}' is not inside a git repository.")

        repo_root = _get_repo_root(target)
        session_id = uuid.uuid4().hex[:8]
        timestamp = int(time.time())
        branch_name = f"{branch_prefix}/task-{timestamp}-{session_id}"
        worktree_dir = self.base_dir / f"session-{session_id}"

        has_head = _has_head_commit(repo_root)

        if has_head:
            cmd = ["git", "worktree", "add", "-b", branch_name, str(worktree_dir), "HEAD"]
        else:
            cmd = ["git", "worktree", "add", "--orphan", "-b", branch_name, str(worktree_dir)]

        proc = subprocess.run(
            cmd,
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Failed creating git worktree for branch '{branch_name}': {proc.stderr.strip()}"
            )

        # If repo had no initial commit, copy current directory contents to prime sandbox
        if not has_head:
            for item in os.listdir(repo_root):
                if item in (".git", "__pycache__", ".venv", "venv"):
                    continue
                src = repo_root / item
                dst = worktree_dir / item
                if src.is_dir():
                    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".git", "__pycache__"))
                else:
                    shutil.copy2(src, dst)

        session = WorktreeSession(
            repo_root=repo_root,
            worktree_dir=worktree_dir,
            branch_name=branch_name,
            has_head=has_head,
        )
        self._active_sessions.append(session)
        return session

    def cleanup_all(self) -> None:
        """Cleans up all active sessions."""
        for session in list(self._active_sessions):
            session.cleanup()
        self._active_sessions.clear()


class DiffViewer:
    """Renders unified diffs and patch reviews with Rich syntax highlighting."""

    @staticmethod
    def render(diff_text: str, console: Console | None = None) -> RenderableType:
        """Returns a Rich panel formatted with syntax-highlighted diff output."""
        if not diff_text.strip():
            return Panel(
                "[dim italic]No modifications detected (clean tree)[/dim italic]",
                title="[bold green]Unified Diff Review[/bold green]",
                border_style="green",
            )

        syntax = Syntax(diff_text, "diff", theme="monokai", line_numbers=True)
        return Panel(
            syntax,
            title="[bold cyan]Sandbox Diff Review (Safe Execution)[/bold cyan]",
            subtitle="[dim]Compare sandbox modifications against main tree[/dim]",
            border_style="cyan",
        )

    @staticmethod
    def render_summary(summary: DiffSummary) -> Table:
        """Returns a concise summary table of changed files and line counts."""
        table = Table(
            title="Sandbox Change Summary",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("File", style="bold white")
        table.add_column("Changes", style="cyan")

        for f in summary.files:
            table.add_row(f, "[dim]Modified[/dim]")

        table.add_row(
            "[bold white]Total[/bold white]",
            f"[bold green]+{summary.insertions}[/bold green] [bold red]-{summary.deletions}[/bold red] ({summary.files_changed} files)",
        )
        return table
