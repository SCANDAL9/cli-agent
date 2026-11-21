"""
Git Worktree Sandboxing and Atomic Diff Engine for cli-agent.
Provides isolated ephemeral branches and worktrees to safeguard the user's active workspace,
with GitHub-style unified diff reviews and atomic patch application.
"""

import ast
import difflib
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tree_sitter_python as tspython
from rich.console import Console, RenderableType
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from tree_sitter import Language, Node, Parser

DEFAULT_WORKTREE_BASE = Path("/tmp/cli-agent/worktrees")
_PY_LANGUAGE = Language(tspython.language())
_IGNORED_WALK_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "node_modules",
}


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

    def create_worker_session(
        self,
        plan_id: str,
        node_id: str,
        base_ref: str = "HEAD",
        repo_path: Path | str | None = None,
        upstream_sessions: list[WorktreeSession] | None = None,
    ) -> WorktreeSession:
        """Creates an isolated per-worker Git worktree at swarm/<plan_id>/<node_id>."""
        target = Path(repo_path or Path.cwd()).resolve()
        if not _is_git_repo(target):
            raise RuntimeError(f"Path '{target}' is not inside a git repository.")

        repo_root = _get_repo_root(target)
        branch_name = f"swarm/{plan_id}/{node_id}"
        worktree_dir = self.base_dir / f"swarm-{plan_id}-{node_id}"

        # Clean up any leftover worktree or branch with the same name
        if worktree_dir.exists():
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree_dir)],
                cwd=repo_root,
                capture_output=True,
                check=False,
            )
            if worktree_dir.exists():
                shutil.rmtree(worktree_dir, ignore_errors=True)
        subprocess.run(
            ["git", "branch", "-D", branch_name],
            cwd=repo_root,
            capture_output=True,
            check=False,
        )

        has_head = _has_head_commit(repo_root)
        if has_head:
            cmd = ["git", "worktree", "add", "-b", branch_name, str(worktree_dir), base_ref]
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
                f"Failed creating worker git worktree for branch '{branch_name}': {proc.stderr.strip()}"
            )

        if not has_head:
            for item in os.listdir(repo_root):
                if item in _IGNORED_WALK_DIRS:
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

        # Pre-apply modifications from completed upstream dependency worktrees
        if upstream_sessions:
            for up_sess in upstream_sessions:
                if not up_sess.worktree_dir.exists():
                    continue
                diff_text = up_sess.get_diff()
                if diff_text.strip():
                    subprocess.run(
                        ["git", "apply", "--3way"],
                        input=diff_text,
                        text=True,
                        cwd=worktree_dir,
                        capture_output=True,
                        check=False,
                    )

        return session

    def cleanup_all(self) -> None:
        """Cleans up all active sessions."""
        for session in list(self._active_sessions):
            session.cleanup()
        self._active_sessions.clear()


@dataclass(slots=True)
class SymbolMergeConflict:
    """Detailed report when two concurrent workers mutate the same AST symbol."""

    file_path: str
    colliding_symbols: list[str]
    worker_a_id: str
    worker_b_id: str
    conflict_diff: str = ""
    base_source: str = ""
    worker_a_source: str = ""
    worker_b_source: str = ""


@dataclass(slots=True)
class MergeResult:
    """Outcome of reconciling multiple worker WorktreeSessions into an integration session."""

    status: str  # "clean" | "ast_resolved" | "conflict" | "error"
    merged_files: list[str] = field(default_factory=list)
    ast_resolved_files: list[str] = field(default_factory=list)
    conflicts: list[SymbolMergeConflict] = field(default_factory=list)
    integration_session: WorktreeSession | None = None
    message: str = ""


@dataclass(slots=True)
class _ParsedPythonAST:
    """Internal representation of top-level and class-level symbols in a Python file."""

    symbols: dict[str, str]
    symbol_spans: dict[str, tuple[int, int]]
    symbol_order: list[str]
    imports: list[str]
    has_error: bool


def _ts_node_text(node: Node | None, source_bytes: bytes) -> str:
    if node is None:
        return ""
    return source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _parse_python_symbols(source: str) -> _ParsedPythonAST:
    """Extracts AST symbols, byte spans, and import lines using Tree-Sitter and ast."""
    source_bytes = source.encode("utf-8")
    parser = Parser(_PY_LANGUAGE)
    tree = parser.parse(source_bytes)
    root = tree.root_node

    has_syntax_error = root.has_error
    if not has_syntax_error:
        try:
            ast.parse(source)
        except SyntaxError:
            has_syntax_error = True

    symbols: dict[str, str] = {}
    symbol_spans: dict[str, tuple[int, int]] = {}
    symbol_order: list[str] = []
    imports: list[str] = []

    for child in root.children:
        node_type = child.type
        if node_type in ("import_statement", "import_from_statement"):
            imp_text = _ts_node_text(child, source_bytes).strip()
            if imp_text and imp_text not in imports:
                imports.append(imp_text)
            continue

        outer_node = child
        inner_node = child
        if node_type == "decorated_definition":
            defn = child.child_by_field_name("definition")
            if defn is not None:
                inner_node = defn

        if inner_node.type == "function_definition":
            name_node = inner_node.child_by_field_name("name")
            sym_name = _ts_node_text(name_node, source_bytes).strip()
            if sym_name:
                symbols[sym_name] = _ts_node_text(outer_node, source_bytes)
                symbol_spans[sym_name] = (outer_node.start_byte, outer_node.end_byte)
                if sym_name not in symbol_order:
                    symbol_order.append(sym_name)

        elif inner_node.type == "class_definition":
            name_node = inner_node.child_by_field_name("name")
            cls_name = _ts_node_text(name_node, source_bytes).strip()
            if not cls_name:
                continue
            body_node = inner_node.child_by_field_name("body")
            method_found = False
            if body_node is not None:
                for member in body_node.children:
                    m_outer = member
                    m_inner = member
                    if member.type == "decorated_definition":
                        m_defn = member.child_by_field_name("definition")
                        if m_defn is not None:
                            m_inner = m_defn
                    if m_inner.type == "function_definition":
                        m_name_node = m_inner.child_by_field_name("name")
                        m_name = _ts_node_text(m_name_node, source_bytes).strip()
                        if m_name:
                            method_found = True
                            qual_name = f"{cls_name}.{m_name}"
                            symbols[qual_name] = _ts_node_text(m_outer, source_bytes)
                            symbol_spans[qual_name] = (m_outer.start_byte, m_outer.end_byte)
                            if qual_name not in symbol_order:
                                symbol_order.append(qual_name)
            if not method_found:
                symbols[cls_name] = _ts_node_text(outer_node, source_bytes)
                symbol_spans[cls_name] = (outer_node.start_byte, outer_node.end_byte)
                if cls_name not in symbol_order:
                    symbol_order.append(cls_name)

        elif node_type == "expression_statement" and child.children:
            first_sub = child.children[0]
            if first_sub.type == "assignment":
                left_node = first_sub.child_by_field_name("left")
                if left_node is not None and left_node.type == "identifier":
                    var_name = _ts_node_text(left_node, source_bytes).strip()
                    if var_name:
                        symbols[var_name] = _ts_node_text(child, source_bytes)
                        symbol_spans[var_name] = (child.start_byte, child.end_byte)
                        if var_name not in symbol_order:
                            symbol_order.append(var_name)

    return _ParsedPythonAST(
        symbols=symbols,
        symbol_spans=symbol_spans,
        symbol_order=symbol_order,
        imports=imports,
        has_error=has_syntax_error,
    )


def _normalize_symbol_body(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


class ASTThreeWayMerger:
    """Reconciles multiple concurrent WorktreeSessions using git merge-file and Tree-Sitter AST."""

    def __init__(self, repo_root: Path | None = None) -> None:
        self.repo_root = (repo_root or Path.cwd()).resolve()

    def _collect_worktree_files(self, root_dir: Path) -> dict[str, bytes]:
        """Collects all tracked/untracked workspace files relative to root_dir."""
        files: dict[str, bytes] = {}
        if not root_dir.exists():
            return files
        for curr_root, dirs, filenames in os.walk(root_dir):
            dirs[:] = [d for d in dirs if d not in _IGNORED_WALK_DIRS]
            for fname in filenames:
                if fname in _IGNORED_WALK_DIRS or fname == ".git":
                    continue
                fpath = Path(curr_root) / fname
                if fpath.is_symlink():
                    continue
                rel = fpath.relative_to(root_dir).as_posix()
                if rel == ".git" or rel.startswith(".git/"):
                    continue
                try:
                    files[rel] = fpath.read_bytes()
                except OSError:
                    continue
        return files

    def _get_worker_deltas(
        self, base_files: dict[str, bytes], worker_files: dict[str, bytes]
    ) -> dict[str, bytes | None]:
        """Returns map of relative file paths changed/added/deleted by a worker."""
        deltas: dict[str, bytes | None] = {}
        all_paths = set(base_files.keys()) | set(worker_files.keys())
        for rel_path in sorted(all_paths):
            in_base = rel_path in base_files
            in_worker = rel_path in worker_files
            if in_base and not in_worker:
                deltas[rel_path] = None  # Deleted
            elif not in_base and in_worker:
                deltas[rel_path] = worker_files[rel_path]  # Added
            elif in_base and in_worker and base_files[rel_path] != worker_files[rel_path]:
                deltas[rel_path] = worker_files[rel_path]  # Modified
        return deltas

    def _run_git_merge_file(
        self, current_text: str, base_text: str, other_text: str
    ) -> tuple[int, str]:
        """Executes `git merge-file --diff3 -p` on temporary files and returns (returncode, merged_text)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            f_curr = tmp / "current"
            f_base = tmp / "base"
            f_other = tmp / "other"
            f_curr.write_text(current_text, encoding="utf-8")
            f_base.write_text(base_text, encoding="utf-8")
            f_other.write_text(other_text, encoding="utf-8")

            proc = subprocess.run(
                ["git", "merge-file", "--diff3", "-p", str(f_curr), str(f_base), str(f_other)],
                cwd=self.repo_root if self.repo_root.exists() else tmp,
                capture_output=True,
                text=True,
                check=False,
            )
            return proc.returncode, proc.stdout

    def _splice_disjoint_python_ast(
        self,
        base_source: str,
        current_source: str,
        other_source: str,
    ) -> str | None:
        """
        Resolves line-level diff3 conflicts (such as EOF append collisions) when AST symbol
        mutations are strictly disjoint by splicing modified/added symbol byte spans.
        """
        base_ast = _parse_python_symbols(base_source)
        curr_ast = _parse_python_symbols(current_source)
        other_ast = _parse_python_symbols(other_source)

        result_source = current_source

        # 1. Union any newly added top-level imports from other_source
        new_imports = [
            imp
            for imp in other_ast.imports
            if imp not in base_ast.imports and imp not in curr_ast.imports
        ]
        if new_imports:
            import_block = "\n".join(new_imports) + "\n"
            result_source = import_block + result_source

        # 2. Apply modified or deleted symbols from other_source relative to base_source
        for sym_name, base_sym_src in base_ast.symbols.items():
            if sym_name not in other_ast.symbols:
                # Deleted in other_source
                refreshed = _parse_python_symbols(result_source)
                if sym_name in refreshed.symbol_spans:
                    start_b, end_b = refreshed.symbol_spans[sym_name]
                    raw_b = result_source.encode("utf-8")
                    result_source = (raw_b[:start_b] + raw_b[end_b:]).decode("utf-8")
            else:
                other_sym_src = other_ast.symbols[sym_name]
                if _normalize_symbol_body(other_sym_src) != _normalize_symbol_body(base_sym_src):
                    # Modified in other_source -> replace span in result_source
                    refreshed = _parse_python_symbols(result_source)
                    if sym_name in refreshed.symbol_spans:
                        start_b, end_b = refreshed.symbol_spans[sym_name]
                        raw_b = result_source.encode("utf-8")
                        replacement_b = other_sym_src.encode("utf-8")
                        result_source = (raw_b[:start_b] + replacement_b + raw_b[end_b:]).decode(
                            "utf-8"
                        )

        # 3. Append newly added symbols from other_source in deterministic topological order
        for sym_name in other_ast.symbol_order:
            if sym_name not in base_ast.symbols:
                refreshed = _parse_python_symbols(result_source)
                if sym_name not in refreshed.symbols:
                    sym_text = other_ast.symbols[sym_name].rstrip()
                    result_source = result_source.rstrip() + "\n\n\n" + sym_text + "\n"

        # 4. Validate merged AST with both ast.parse() and Tree-Sitter
        validated = _parse_python_symbols(result_source)
        if validated.has_error:
            return None
        return result_source

    def merge_worker_sessions(
        self,
        base_session: WorktreeSession | Path,
        worker_sessions: dict[str, WorktreeSession],
        integration_session: WorktreeSession,
    ) -> MergeResult:
        """
        Reconciles multiple concurrent worker WorktreeSessions into integration_session:
        - Stage 1: Disjoint modified files are fast-forward copied/deleted into integration_session.
        - Stage 2: Shared modified files undergo Tree-Sitter AST SymbolDelta intersection check:
          * Disjoint symbols -> 3-way merge (`git merge-file --diff3 -p`) + AST splicing fallback.
          * Overlapping symbol mutations -> reject and flag `SymbolMergeConflict`.
        """
        base_dir = (
            base_session.worktree_dir
            if isinstance(base_session, WorktreeSession)
            else Path(base_session).resolve()
        )
        int_dir = integration_session.worktree_dir
        if not int_dir.exists():
            return MergeResult(
                status="error",
                integration_session=integration_session,
                message="Integration worktree directory does not exist.",
            )

        base_files = self._collect_worktree_files(base_dir)
        file_to_workers: dict[str, list[tuple[str, bytes | None]]] = {}

        for worker_id, w_sess in worker_sessions.items():
            w_files = self._collect_worktree_files(w_sess.worktree_dir)
            deltas = self._get_worker_deltas(base_files, w_files)
            for rel_path, new_bytes in deltas.items():
                file_to_workers.setdefault(rel_path, []).append((worker_id, new_bytes))

        merged_files: list[str] = []
        ast_resolved_files: list[str] = []
        conflicts: list[SymbolMergeConflict] = []

        for rel_path in sorted(file_to_workers.keys()):
            worker_edits = file_to_workers[rel_path]
            target_file = int_dir / rel_path

            # Stage 1: Disjoint file modified by exactly 1 worker
            if len(worker_edits) == 1:
                _, content_bytes = worker_edits[0]
                if content_bytes is None:
                    if target_file.exists():
                        target_file.unlink()
                else:
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    target_file.write_bytes(content_bytes)
                merged_files.append(rel_path)
                continue

            # Stage 2: Shared file modified by >= 2 workers
            base_bytes = base_files.get(rel_path, b"")
            base_text = base_bytes.decode("utf-8", errors="replace")
            is_python = rel_path.endswith(".py")
            base_ast = _parse_python_symbols(base_text) if is_python else None

            first_worker_id, first_bytes = worker_edits[0]
            current_worker_id = first_worker_id
            current_text = (first_bytes or b"").decode("utf-8", errors="replace")
            file_had_conflict = False
            file_ast_resolved = False

            for next_worker_id, next_bytes in worker_edits[1:]:
                next_text = (next_bytes or b"").decode("utf-8", errors="replace")

                # Identical edit fast-path
                if current_text == next_text:
                    continue

                if is_python and base_ast is not None:
                    curr_ast = _parse_python_symbols(current_text)
                    next_ast = _parse_python_symbols(next_text)

                    curr_modified = {
                        s
                        for s, b_src in base_ast.symbols.items()
                        if s not in curr_ast.symbols
                        or _normalize_symbol_body(curr_ast.symbols[s])
                        != _normalize_symbol_body(b_src)
                    }
                    next_modified = {
                        s
                        for s, b_src in base_ast.symbols.items()
                        if s not in next_ast.symbols
                        or _normalize_symbol_body(next_ast.symbols[s])
                        != _normalize_symbol_body(b_src)
                    }

                    curr_added = set(curr_ast.symbols.keys()) - set(base_ast.symbols.keys())
                    next_added = set(next_ast.symbols.keys()) - set(base_ast.symbols.keys())

                    colliding_modified = {
                        s
                        for s in (curr_modified & next_modified)
                        if _normalize_symbol_body(curr_ast.symbols.get(s, ""))
                        != _normalize_symbol_body(next_ast.symbols.get(s, ""))
                    }
                    colliding_added = {
                        s
                        for s in (curr_added & next_added)
                        if _normalize_symbol_body(curr_ast.symbols[s])
                        != _normalize_symbol_body(next_ast.symbols[s])
                    }
                    colliding = sorted(colliding_modified | colliding_added)

                    if colliding:
                        first_sym = colliding[0]
                        diff_lines = list(
                            difflib.unified_diff(
                                current_text.splitlines(keepends=True),
                                next_text.splitlines(keepends=True),
                                fromfile=f"{current_worker_id}/{rel_path}",
                                tofile=f"{next_worker_id}/{rel_path}",
                            )
                        )
                        conflicts.append(
                            SymbolMergeConflict(
                                file_path=rel_path,
                                colliding_symbols=colliding,
                                worker_a_id=current_worker_id,
                                worker_b_id=next_worker_id,
                                conflict_diff="".join(diff_lines),
                                base_source=base_ast.symbols.get(first_sym, ""),
                                worker_a_source=curr_ast.symbols.get(first_sym, ""),
                                worker_b_source=next_ast.symbols.get(first_sym, ""),
                            )
                        )
                        file_had_conflict = True
                        break

                # Disjoint symbols (or non-Python file): run 3-way git merge-file
                ret_code, merged_out = self._run_git_merge_file(current_text, base_text, next_text)
                if ret_code == 0 and "<<<<<<<" not in merged_out:
                    if is_python:
                        merged_ast = _parse_python_symbols(merged_out)
                        if not merged_ast.has_error:
                            current_text = merged_out
                            continue
                    else:
                        current_text = merged_out
                        continue

                # Line-level diff3 conflict occurred on disjoint AST symbols -> splice AST
                if is_python:
                    spliced = self._splice_disjoint_python_ast(base_text, current_text, next_text)
                    if spliced is not None:
                        current_text = spliced
                        file_ast_resolved = True
                        continue

                # Unresolvable conflict
                conflicts.append(
                    SymbolMergeConflict(
                        file_path=rel_path,
                        colliding_symbols=["<file-level-conflict>"],
                        worker_a_id=current_worker_id,
                        worker_b_id=next_worker_id,
                        conflict_diff=merged_out,
                        base_source=base_text,
                        worker_a_source=current_text,
                        worker_b_source=next_text,
                    )
                )
                file_had_conflict = True
                break

            if not file_had_conflict:
                target_file.parent.mkdir(parents=True, exist_ok=True)
                target_file.write_text(current_text, encoding="utf-8")
                merged_files.append(rel_path)
                if file_ast_resolved:
                    ast_resolved_files.append(rel_path)

        if conflicts:
            return MergeResult(
                status="conflict",
                merged_files=merged_files,
                ast_resolved_files=ast_resolved_files,
                conflicts=conflicts,
                integration_session=integration_session,
                message=f"Detected {len(conflicts)} AST symbol merge conflict(s).",
            )

        status = "ast_resolved" if ast_resolved_files else "clean"
        return MergeResult(
            status=status,
            merged_files=merged_files,
            ast_resolved_files=ast_resolved_files,
            conflicts=[],
            integration_session=integration_session,
            message=f"Successfully merged {len(merged_files)} file(s) ({status}).",
        )


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
