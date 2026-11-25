"""Isolated Git worktrees used as workflow checkpoints and rollback boundaries."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .models import WorkflowValidationError


def _run_git(root: Path, *arguments: str, input_bytes: bytes | None = None) -> bytes:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        input=input_bytes,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise WorkflowValidationError(f"Git checkpoint command failed: {message}")
    return result.stdout


@dataclass
class WorkflowCheckpoint:
    repository_root: Path
    worktree_path: Path
    branch_name: str
    tag_name: str
    temporary_parent: Path
    worktree_added: bool = False
    branch_created: bool = False
    tag_created: bool = False
    retained: bool = False
    closed: bool = False

    def discard(self) -> None:
        """Remove only this run's worktree and uniquely generated branch."""
        if self.closed:
            return
        if self.worktree_added:
            worktree_result = subprocess.run(
                ["git", "worktree", "remove", "--force", str(self.worktree_path)],
                cwd=self.repository_root,
                capture_output=True,
                check=False,
            )
            if worktree_result.returncode != 0:
                detail = worktree_result.stderr.decode("utf-8", errors="replace").strip()
                raise WorkflowValidationError(
                    f"Could not remove workflow worktree '{self.worktree_path}': {detail}"
                )
            self.worktree_added = False
        if self.branch_created:
            branch_result = subprocess.run(
                ["git", "branch", "-D", self.branch_name],
                cwd=self.repository_root,
                capture_output=True,
                check=False,
            )
            if branch_result.returncode != 0:
                detail = branch_result.stderr.decode("utf-8", errors="replace").strip()
                raise WorkflowValidationError(
                    f"Could not remove workflow branch '{self.branch_name}': {detail}"
                )
            self.branch_created = False
        if self.tag_created:
            tag_result = subprocess.run(
                ["git", "tag", "-d", self.tag_name],
                cwd=self.repository_root,
                capture_output=True,
                check=False,
            )
            if tag_result.returncode != 0:
                detail = tag_result.stderr.decode("utf-8", errors="replace").strip()
                raise WorkflowValidationError(
                    f"Could not remove workflow checkpoint tag '{self.tag_name}': {detail}"
                )
            self.tag_created = False
        shutil.rmtree(self.temporary_parent, ignore_errors=True)
        self.closed = True

    def keep(self) -> None:
        """Leave a successful worktree/branch for inspection and manual integration."""
        self.retained = True
        self.closed = True


class WorkflowCheckpointManager:
    """Creates an isolated worktree, including tracked and visible untracked edits."""

    def __init__(self, repository_root: Path | str | None = None) -> None:
        requested_root = Path(repository_root or Path.cwd()).resolve()
        root = _run_git(requested_root, "rev-parse", "--show-toplevel").decode().strip()
        self.repository_root = Path(root).resolve()

    def create(self, *, run_id: str, branch_prefix: str = "cli-agent-wf-") -> WorkflowCheckpoint:
        safe_run_id = "".join(char for char in run_id.lower() if char.isalnum())[:16]
        safe_prefix = "".join(
            char if char.isalnum() or char in "-_" else "-" for char in branch_prefix
        )
        safe_prefix = safe_prefix.strip("-_") or "cli-agent-wf"
        tag_name = f"{safe_prefix}-{safe_run_id}"
        branch_name = f"{tag_name}-branch"
        temporary_parent = Path(tempfile.mkdtemp(prefix="cli-agent-workflow-"))
        worktree_path = temporary_parent / "workspace"
        checkpoint = WorkflowCheckpoint(
            repository_root=self.repository_root,
            worktree_path=worktree_path,
            branch_name=branch_name,
            tag_name=tag_name,
            temporary_parent=temporary_parent,
        )
        try:
            _run_git(self.repository_root, "tag", tag_name, "HEAD")
            checkpoint.tag_created = True
            _run_git(
                self.repository_root,
                "worktree",
                "add",
                "-b",
                branch_name,
                str(worktree_path),
                "HEAD",
            )
            checkpoint.worktree_added = True
            checkpoint.branch_created = True
            self._copy_tracked_changes(worktree_path)
            self._copy_untracked_files(worktree_path)
            return checkpoint
        except BaseException as exc:
            try:
                checkpoint.discard()
            except Exception as cleanup_error:
                raise WorkflowValidationError(
                    f"Checkpoint creation failed: {exc}; cleanup also failed: {cleanup_error}"
                ) from cleanup_error
            raise

    def _copy_tracked_changes(self, worktree_path: Path) -> None:
        patch = _run_git(self.repository_root, "diff", "--binary", "HEAD", "--")
        if not patch:
            return
        _run_git(worktree_path, "apply", "--binary", "--check", "-", input_bytes=patch)
        _run_git(worktree_path, "apply", "--binary", "-", input_bytes=patch)

    def _copy_untracked_files(self, worktree_path: Path) -> None:
        output = _run_git(self.repository_root, "ls-files", "--others", "--exclude-standard", "-z")
        for encoded_path in output.split(b"\0"):
            if not encoded_path:
                continue
            relative = Path(encoded_path.decode("utf-8", errors="surrogateescape"))
            if relative.is_absolute() or ".." in relative.parts:
                raise WorkflowValidationError("Git returned an unsafe untracked path.")
            source = self.repository_root / relative
            resolved_source = source.resolve(strict=False)
            if not resolved_source.is_relative_to(self.repository_root):
                raise WorkflowValidationError(
                    f"Untracked path '{relative}' resolves outside the repository; refusing to copy it."
                )
            target = worktree_path / relative
            if source.is_symlink():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(source.readlink())
            elif source.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
