"""
Local system and memory tools for the Gemini agent.
Enables file reading, directory exploration, code searching, and persistent SQLite memories.
"""

import contextvars
import hashlib
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .code_graph import SUPPORTED_EXTENSIONS, CodeGraphIndexer, CodeGraphStore
from .memory import MemoryStore, resolve_project_key
from .sandbox import (
    SandboxUnavailableError,
    build_bwrap_argv,
    build_bwrap_probe_argv,
    reduced_shell_argv,
    sandbox_environment,
)

memory_store = MemoryStore()
code_graph_store = CodeGraphStore()
code_graph_indexer = CodeGraphIndexer(code_graph_store)

_active_sandbox_var: ContextVar[Path | None] = ContextVar("_active_sandbox_var", default=None)
_allowed_files_var: ContextVar[frozenset[str] | None] = ContextVar(
    "_allowed_files_var", default=None
)
_active_worker_id_var: ContextVar[str | None] = ContextVar("_active_worker_id_var", default=None)
_allowed_write_paths_var: contextvars.ContextVar[set[Path] | None] = contextvars.ContextVar(
    "_allowed_write_paths_var", default=None
)


@dataclass
class BackgroundProcessEntry:
    job_id: str
    command: str
    cwd: Path
    process: subprocess.Popen[str]
    log_path: Path
    started_at: float
    sandbox_temp_dir: Path | None = None


_background_processes: dict[str, BackgroundProcessEntry] = {}


class ToolScopeViolationError(PermissionError):
    """Raised when a scoped worker attempts to mutate a file outside its allowed_files ACL."""

    def __init__(
        self,
        message: str,
        *,
        violation_type: str = "ALLOWED_FILES_ACL_VIOLATION",
        attempted_path: str = "",
        allowed_files: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.violation_type = violation_type
        self.attempted_path = attempted_path
        self.allowed_files: list[str] = allowed_files if allowed_files is not None else []

    def to_dict(self) -> dict[str, Any]:
        """Serializes the violation into a structured tool response dictionary."""
        return {
            "error": str(self),
            "violation_type": self.violation_type,
            "attempted_path": self.attempted_path,
            "allowed_files": self.allowed_files,
        }


@contextmanager
def worker_execution_scope(
    sandbox_dir: Path | None,
    allowed_files: list[str] | None = None,
    worker_id: str | None = None,
) -> Iterator[None]:
    """Binds a coroutine-local Git worktree sandbox and hard allowed_files ACL."""
    resolved_sandbox = sandbox_dir.resolve() if sandbox_dir else None
    normalized_allowed: frozenset[str] | None = None
    if allowed_files is not None:
        normalized_allowed = frozenset(Path(f).as_posix().lstrip("./") for f in allowed_files)

    tok_sandbox = _active_sandbox_var.set(resolved_sandbox)
    tok_allowed = _allowed_files_var.set(normalized_allowed)
    tok_worker = _active_worker_id_var.set(worker_id)
    try:
        yield
    finally:
        _active_sandbox_var.reset(tok_sandbox)
        _allowed_files_var.reset(tok_allowed)
        _active_worker_id_var.reset(tok_worker)


def set_memory_store(store: MemoryStore) -> None:
    """Configures the active MemoryStore instance used by memory tools."""
    global memory_store
    memory_store = store


def set_code_graph_store(store: CodeGraphStore) -> None:
    """Configures the active CodeGraphStore and indexer used by AST tools."""
    global code_graph_store, code_graph_indexer
    code_graph_store = store
    code_graph_indexer = CodeGraphIndexer(store)


def set_active_sandbox(sandbox_dir: Path | str | None) -> None:
    """Sets or clears the active ephemeral sandbox directory for file and command operations."""
    resolved = Path(sandbox_dir).resolve() if sandbox_dir else None
    _active_sandbox_var.set(resolved)


def get_active_sandbox() -> Path | None:
    """Returns the current active sandbox directory, or None if in direct workspace mode."""
    return _active_sandbox_var.get()


def set_allowed_write_paths(paths: list[str | Path] | None) -> contextvars.Token[set[Path] | None]:
    """Sets the context-local allowed write paths for scoped workers."""
    if paths is None:
        return _allowed_write_paths_var.set(None)
    resolved_set = {_resolve_tool_path(str(p)) for p in paths if str(p).strip()}
    return _allowed_write_paths_var.set(resolved_set)


def reset_allowed_write_paths(token: contextvars.Token[set[Path] | None]) -> None:
    """Resets the context-local allowed write paths."""
    _allowed_write_paths_var.reset(token)


def get_allowed_write_paths() -> set[Path] | None:
    """Returns the active context-local allowed write paths, or None if unrestricted."""
    return _allowed_write_paths_var.get()


def _check_write_allowed(resolved_path: Path) -> str | None:
    """Returns an error message if resolved_path violates active allowed_write_paths."""
    allowed = _allowed_write_paths_var.get()
    if allowed is not None and len(allowed) > 0 and resolved_path not in allowed:
        allowed_strs = sorted(str(p) for p in allowed)
        return (
            f"Access denied: Modifying '{resolved_path}' is outside your approved file scope. "
            f"Allowed files: {allowed_strs}"
        )
    return None


def _resolve_tool_path(raw_path: str) -> Path:
    """Resolves a path, remapping relative or workspace paths into the active sandbox when enabled."""
    p = Path(raw_path).expanduser()
    active_sandbox = _active_sandbox_var.get()
    if active_sandbox:
        if not p.is_absolute():
            return (active_sandbox / p).resolve()
        try:
            p.resolve().relative_to(active_sandbox.resolve())
            return p.resolve()
        except ValueError:
            pass
        try:
            rel = p.resolve().relative_to(Path.cwd().resolve())
            return (active_sandbox / rel).resolve()
        except ValueError:
            return p.resolve()
    return p.resolve()


def _enforce_write_scope(raw_path: str, resolved_path: Path) -> None:
    """Verifies that resolved_path is inside the active sandbox and matches allowed_files."""
    sandbox_root = _active_sandbox_var.get()
    allowed_set = _allowed_files_var.get()
    worker_id = _active_worker_id_var.get()
    base_root = sandbox_root.resolve() if sandbox_root is not None else Path.cwd().resolve()

    resolved_target = resolved_path.resolve()
    is_scoped = sandbox_root is not None or allowed_set is not None or worker_id is not None

    # 1. Path Traversal Guard (Prevent '../../etc/passwd' or symlink escape)
    try:
        rel_posix = resolved_target.relative_to(base_root).as_posix()
    except ValueError as exc:
        if not is_scoped and ".." not in Path(raw_path).parts:
            temp_root = Path(tempfile.gettempdir()).resolve()
            try:
                resolved_target.relative_to(temp_root)
                return
            except ValueError:
                pass
        raise ToolScopeViolationError(
            f"Security Violation: Path '{raw_path}' escapes workspace root '{base_root}'.",
            violation_type="PATH_TRAVERSAL_VIOLATION",
            attempted_path=raw_path,
            allowed_files=sorted(allowed_set) if allowed_set is not None else [],
        ) from exc

    # 2. Hard allowed_files ACL Guard
    if allowed_set is not None and rel_posix not in allowed_set:
        effective_worker = worker_id or "scoped-worker"
        allowed_list = ", ".join(sorted(allowed_set)) or "<none (read-only)>"
        raise ToolScopeViolationError(
            f"Access Denied for worker [{effective_worker}]: Attempted to modify '{rel_posix}', "
            f"which is outside approved allowed_files: [{allowed_list}].",
            violation_type="ALLOWED_FILES_ACL_VIOLATION",
            attempted_path=rel_posix,
            allowed_files=sorted(allowed_set),
        )


def read_file(path: str, start_line: int = 1, end_line: int = 300) -> dict[str, Any]:
    """Reads a section of lines from a local file."""
    try:
        resolved_path = _resolve_tool_path(path)
        if not resolved_path.exists():
            return {"error": f"File '{path}' does not exist."}
        if not resolved_path.is_file():
            return {"error": f"Path '{path}' is not a file."}

        with open(resolved_path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        total_lines = len(lines)
        start_idx = max(0, start_line - 1)
        end_idx = min(total_lines, end_line)

        selected_lines = lines[start_idx:end_idx]
        formatted = "".join(
            f"{i + start_idx + 1:4d} | {line}" for i, line in enumerate(selected_lines)
        )

        return {
            "path": str(resolved_path),
            "total_lines": total_lines,
            "start_line": start_line,
            "end_line": end_idx,
            "content": formatted,
        }
    except Exception as e:
        return {"error": f"Failed to read '{path}': {str(e)}"}


def list_directory(path: str = ".", max_depth: int = 2) -> dict[str, Any]:
    """Lists files and directories under a given path up to a specified depth."""
    try:
        base_path = _resolve_tool_path(path)
        if not base_path.exists():
            return {"error": f"Directory '{path}' does not exist."}
        if not base_path.is_dir():
            return {"error": f"Path '{path}' is not a directory."}

        from . import fast_tools

        native_result = fast_tools.fast_list_directory(str(base_path), max_depth=max_depth)
        if native_result is not None:
            return native_result

        entries: list[str] = []
        ignored_names = {".git", ".venv", "node_modules", "__pycache__", ".next", "dist", "build"}

        for root, dirs, files in os.walk(base_path):
            dirs[:] = [d for d in dirs if d not in ignored_names and not d.startswith(".")]
            rel_path = Path(root).relative_to(base_path)
            depth = len(rel_path.parts)
            if depth >= max_depth:
                dirs.clear()

            indent = "  " * depth
            current_folder = rel_path.name or base_path.name
            if depth > 0:
                entries.append(f"{indent[:-2]}📁 {current_folder}/")

            for f in sorted(files)[:50]:
                if not f.startswith("."):
                    entries.append(f"{indent}📄 {f}")

            if len(entries) > 200:
                entries.append(f"{indent}... [truncated list of files]")
                break

        return {"path": str(base_path), "tree": "\n".join(entries)}
    except Exception as e:
        return {"error": f"Failed to list directory '{path}': {str(e)}"}


def search_code(pattern: str, path: str = ".", file_pattern: str = "*") -> dict[str, Any]:
    """Searches for regex/text pattern matches inside files."""
    try:
        base_path = _resolve_tool_path(path)
        if not base_path.exists():
            return {"error": f"Path '{path}' does not exist."}

        from . import fast_tools

        native_result = fast_tools.fast_search_code(
            pattern, path=str(base_path), file_pattern=file_pattern
        )
        if native_result is not None:
            return native_result

        regex = re.compile(pattern, re.IGNORECASE)
        results = []
        ignored_dirs = {".git", ".venv", "node_modules", "__pycache__", ".next", "dist", "build"}

        for root, dirs, files in os.walk(base_path):
            dirs[:] = [d for d in dirs if d not in ignored_dirs and not d.startswith(".")]
            for file_name in files:
                if file_name.startswith("."):
                    continue
                file_path = Path(root) / file_name
                try:
                    with open(file_path, encoding="utf-8", errors="ignore") as f:
                        for lineno, line in enumerate(f, 1):
                            if regex.search(line):
                                rel_path = file_path.relative_to(base_path)
                                results.append(f"{rel_path}:{lineno}: {line.strip()[:150]}")
                                if len(results) >= 50:
                                    break
                except Exception:
                    continue
            if len(results) >= 50:
                break

        return {
            "pattern": pattern,
            "total_matches": len(results),
            "matches": results or ["No matches found."],
        }
    except Exception as e:
        return {"error": f"Failed searching code for '{pattern}': {str(e)}"}


def lsp_workspace_diagnostics(
    path: str = ".", max_files: int = 250, timeout_ms: int = 50
) -> dict[str, Any]:
    """Collect bounded Python LSP/AST diagnostics without changing workspace files."""
    try:
        if isinstance(max_files, bool) or not isinstance(max_files, int) or max_files < 1:
            return {"status": "failed", "exit_code": 1, "error": "max_files must be positive."}
        if (
            isinstance(timeout_ms, bool)
            or not isinstance(timeout_ms, int)
            or not 1 <= timeout_ms <= 1000
        ):
            return {
                "status": "failed",
                "exit_code": 1,
                "error": "timeout_ms must be between 1 and 1000.",
            }
        base_path = _resolve_tool_path(path)
        if not base_path.exists():
            return {"status": "failed", "exit_code": 1, "error": f"Path '{path}' does not exist."}
        if base_path.is_file() and base_path.suffix != ".py":
            return {
                "status": "failed",
                "exit_code": 1,
                "error": "LSP diagnostics require Python files.",
            }

        candidates: list[Path] = []
        if base_path.is_file():
            candidates = [base_path]
            truncated = False
        else:
            truncated = False
            ignored = {".git", ".venv", "venv", "node_modules", "__pycache__", "dist", "build"}
            for root, directories, filenames in os.walk(base_path):
                directories[:] = sorted(
                    name for name in directories if name not in ignored and not name.startswith(".")
                )
                for filename in sorted(filenames):
                    if not filename.endswith(".py"):
                        continue
                    candidates.append(Path(root) / filename)
                    if len(candidates) > min(max_files, 1000):
                        truncated = True
                        break
                if truncated:
                    break
            candidates = candidates[: min(max_files, 1000)]

        from .lsp import get_lsp_manager

        manager = get_lsp_manager(base_path if base_path.is_dir() else base_path.parent)
        reports = []
        scan_errors: list[str] = []
        for file_path in candidates:
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
                if len(content.encode("utf-8")) > 1_000_000:
                    scan_errors.append(f"{file_path}: file exceeds the 1 MB diagnostic limit")
                    continue
                report = manager.check_file_sync(file_path, content, timeout_ms=timeout_ms)
                reports.append(report)
            except OSError as exc:
                scan_errors.append(f"{file_path}: {exc}")

        diagnostics = [
            {"file_path": report.file_path, **diagnostic.to_dict()}
            for report in reports
            for diagnostic in report.diagnostics
        ][:500]
        error_count = sum(report.error_count for report in reports)
        warning_count = sum(report.warning_count for report in reports)
        failed = error_count > 0 or bool(scan_errors)
        return {
            "status": "failed" if failed else "success",
            "exit_code": 1 if failed else 0,
            "checked_files": len(reports),
            "truncated": truncated,
            "error_count": error_count,
            "warning_count": warning_count,
            "scan_errors": scan_errors,
            "diagnostics": diagnostics,
        }
    except Exception as exc:
        return {"status": "failed", "exit_code": 1, "error": str(exc)}


def ast_grep_replace(
    pattern: str,
    replacement: str,
    language: str,
    path: str = ".",
) -> dict[str, Any]:
    """Previews an AST-aware rewrite without applying changes to project files."""
    from . import fast_tools

    return fast_tools.ast_grep_replace(pattern, replacement, language, path)


async def hybrid_search_code(query: str, limit: int = 5) -> dict[str, Any]:
    """Performs hybrid code search combining sparse keyword matching (BM25/FTS5) with dense semantic vector embeddings via Reciprocal Rank Fusion (RRF)."""
    try:
        from .search import get_hybrid_search_engine

        engine = get_hybrid_search_engine()
        results = await engine.search(query=query, limit=limit)
        return {
            "query": query,
            "count": len(results),
            "results": results or [{"message": "No relevant code chunks found."}],
        }
    except Exception as e:
        return {"error": f"Failed executing hybrid code search: {str(e)}"}


async def semantic_search_code(query: str, limit: int = 5) -> dict[str, Any]:
    """Searches code intent with the local dense-and-sparse hybrid index."""
    return await hybrid_search_code(query, limit)


def save_memory(
    title: str,
    content: str,
    memory_type: str = "discovery",
    scope: str = "project",
    project_key: str | None = None,
) -> dict[str, Any]:
    """Saves a key fact, decision, or discovery into persistent SQLite memory with project scoping."""
    mem_id = memory_store.save_memory(
        title=title,
        content=content,
        memory_type=memory_type,
        project_key=project_key,
        scope=scope,
    )
    return {
        "status": "saved",
        "id": mem_id,
        "title": title,
        "type": memory_type,
        "scope": scope,
        "message": f"Successfully persisted memory #{mem_id} into SQLite.",
    }


def search_memory(query: str, project_key: str | None = None) -> dict[str, Any]:
    """Searches past memories, notes, and session decisions from SQLite for the active workspace."""
    memories = memory_store.search_memory(query=query, project_key=project_key, limit=5)
    return {
        "query": query,
        "count": len(memories),
        "results": memories or [{"message": "No relevant memory found for this query."}],
    }


def write_file(path: str, content: str, overwrite: bool = False) -> dict[str, Any]:
    """Writes full content to a file, creating parent directories if necessary."""
    try:
        resolved = _resolve_tool_path(path)
        err = _check_write_allowed(resolved)
        if err:
            return {"error": err}
        _enforce_write_scope(path, resolved)
        if resolved.exists() and not overwrite:
            return {"error": f"File '{path}' already exists. Pass overwrite=True to replace it."}

        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")

        if resolved.suffix in SUPPORTED_EXTENSIONS:
            try:
                code_graph_indexer.index_file(resolved)
            except Exception:
                pass

        resp: dict[str, Any] = {
            "status": "success",
            "path": str(resolved),
            "bytes_written": len(content.encode("utf-8")),
            "lines": len(content.splitlines()),
        }

        # W5.3 Headless LSP diagnostic enrichment
        if resolved.suffix == ".py":
            try:
                from .lsp import get_lsp_manager

                lsp_mgr = get_lsp_manager()
                lsp_report = lsp_mgr.check_file_sync(resolved, content, timeout_ms=50.0)
                resp["lsp_status"] = "clean" if lsp_report.is_clean else "diagnostics_found"
                resp["lsp_errors"] = lsp_report.error_count
                resp["lsp_warnings"] = lsp_report.warning_count
                resp["lsp_latency_ms"] = lsp_report.latency_ms
                resp["lsp_diagnostics"] = [d.to_dict() for d in lsp_report.diagnostics[:20]]
                if lsp_report.error_count > 0:
                    resp["lsp_hint"] = (
                        f"LSP detected {lsp_report.error_count} error(s) in '{path}'. "
                        "Fix these line-level diagnostics immediately via edit_file before running pytest."
                    )
            except Exception:
                pass

        return resp
    except ToolScopeViolationError as e:
        return e.to_dict()
    except Exception as e:
        return {"error": f"Failed writing to '{path}': {str(e)}"}


def edit_file(path: str, target: str, replacement: str) -> dict[str, Any]:
    """Replaces a unique occurrence of target string with replacement, with whitespace-normalized fallback."""
    try:
        resolved = _resolve_tool_path(path)
        err = _check_write_allowed(resolved)
        if err:
            return {"error": err}
        _enforce_write_scope(path, resolved)
        if not resolved.is_file():
            return {"error": f"Path '{path}' is not an existing file."}

        with open(resolved, encoding="utf-8", newline="") as f:
            content = f.read()

        if not target:
            return {"error": "Target substring was not found in the file."}

        occurrences = content.count(target)
        if occurrences > 1:
            return {
                "error": f"Target substring occurs {occurrences} times. Must be uniquely identifiable."
            }

        if occurrences == 1:
            new_content = content.replace(target, replacement, 1)
        else:
            # Fallback 1: Normalize CRLF vs LF differences
            norm_content = content.replace("\r\n", "\n")
            norm_target = target.replace("\r\n", "\n")
            norm_replacement = replacement.replace("\r\n", "\n")

            norm_occurrences = norm_content.count(norm_target)
            if norm_occurrences > 1:
                return {
                    "error": f"Target substring occurs {norm_occurrences} times. Must be uniquely identifiable."
                }
            if norm_occurrences == 1:
                new_content = norm_content.replace(norm_target, norm_replacement, 1)
            else:
                # Fallback 2: Whitespace-normalized matching (ignoring leading/trailing indentation)
                content_lines = norm_content.splitlines(keepends=True)
                target_lines = norm_target.splitlines()
                while target_lines and not target_lines[0].strip():
                    target_lines.pop(0)
                while target_lines and not target_lines[-1].strip():
                    target_lines.pop()

                if not target_lines:
                    return {"error": "Target substring was not found in the file."}

                m = len(target_lines)
                stripped_target = [line.strip() for line in target_lines]
                match_indices: list[int] = []

                for i in range(len(content_lines) - m + 1):
                    if all(content_lines[i + k].strip() == stripped_target[k] for k in range(m)):
                        match_indices.append(i)

                if len(match_indices) == 0:
                    return {"error": "Target substring was not found in the file."}
                if len(match_indices) > 1:
                    return {
                        "error": f"Target substring occurs {len(match_indices)} times. Must be uniquely identifiable."
                    }

                idx = match_indices[0]
                matched_slice = "".join(content_lines[idx : idx + m])
                rep_text = norm_replacement
                if matched_slice.endswith("\n") and not rep_text.endswith("\n"):
                    rep_text += "\n"
                new_content = (
                    "".join(content_lines[:idx]) + rep_text + "".join(content_lines[idx + m :])
                )

        resolved.write_text(new_content, encoding="utf-8")

        if resolved.suffix in SUPPORTED_EXTENSIONS:
            try:
                code_graph_indexer.index_file(resolved)
            except Exception:
                pass

        resp: dict[str, Any] = {
            "status": "success",
            "path": str(resolved),
            "replacements_made": 1,
        }

        # W5.3 Headless LSP diagnostic enrichment
        if resolved.suffix == ".py":
            try:
                from .lsp import get_lsp_manager

                lsp_mgr = get_lsp_manager()
                lsp_report = lsp_mgr.check_file_sync(resolved, new_content, timeout_ms=50.0)
                resp["lsp_status"] = "clean" if lsp_report.is_clean else "diagnostics_found"
                resp["lsp_errors"] = lsp_report.error_count
                resp["lsp_warnings"] = lsp_report.warning_count
                resp["lsp_latency_ms"] = lsp_report.latency_ms
                resp["lsp_diagnostics"] = [d.to_dict() for d in lsp_report.diagnostics[:20]]
                if lsp_report.error_count > 0:
                    resp["lsp_hint"] = (
                        f"LSP detected {lsp_report.error_count} error(s) in '{path}'. "
                        "Fix these line-level diagnostics immediately via edit_file before running pytest."
                    )
            except Exception:
                pass

        return resp
    except ToolScopeViolationError as e:
        return e.to_dict()
    except Exception as e:
        return {"error": f"Failed editing '{path}': {str(e)}"}


def get_symbol_outline(file_path: str) -> dict[str, Any]:
    """Extracts a structural outline of a source file using Tree-sitter without line-by-line bodies."""
    try:
        resolved = _resolve_tool_path(file_path)
        if not resolved.exists():
            return {"error": f"File '{file_path}' does not exist."}
        if not resolved.is_file():
            return {"error": f"Path '{file_path}' is not a file."}

        project_key = resolve_project_key()
        code_graph_indexer.index_file(resolved, project_key)
        outline = code_graph_store.get_outline(project_key, str(resolved))
        return {
            "file_path": str(resolved),
            "outline": outline,
        }
    except Exception as e:
        return {"error": f"Failed generating symbol outline: {str(e)}"}


def find_symbol_definition(symbol_name: str) -> dict[str, Any]:
    """Finds definitions of a symbol (classes, functions, methods) across the workspace."""
    try:
        project_key = resolve_project_key()
        results = code_graph_store.find_definition(project_key, symbol_name)
        return {
            "symbol_name": symbol_name,
            "project_key": project_key,
            "definitions": results,
            "count": len(results),
        }
    except Exception as e:
        return {"error": f"Failed searching symbol definition: {str(e)}"}


def find_symbol_references(symbol_name: str) -> dict[str, Any]:
    """Finds references, invocations, inheritances, or imports of a symbol across the project."""
    try:
        project_key = resolve_project_key()
        results = code_graph_store.find_references(project_key, symbol_name)
        return {
            "symbol_name": symbol_name,
            "project_key": project_key,
            "references": results,
            "count": len(results),
        }
    except Exception as e:
        return {"error": f"Failed searching symbol references: {str(e)}"}


def analyze_symbol_impact(symbol_name: str) -> dict[str, Any]:
    """Analyzes the structural impact of modifying a symbol (definitions, callers, dependent files, risk rating)."""
    try:
        project_key = resolve_project_key()
        return code_graph_store.get_symbol_impact(project_key, symbol_name)
    except Exception as e:
        return {"error": f"Failed analyzing symbol impact: {str(e)}"}


def _validate_graph_query(symbol_name: str, max_depth: int) -> str | None:
    if not isinstance(symbol_name, str) or not symbol_name.strip():
        return "symbol_name must be a non-empty string."
    if isinstance(max_depth, bool) or not isinstance(max_depth, int) or not 1 <= max_depth <= 5:
        return "max_depth must be an integer between 1 and 5."
    return None


def get_symbol_call_tree(symbol_name: str, max_depth: int = 3) -> dict[str, Any]:
    """Returns bounded upstream callers and downstream callees for a symbol."""
    error = _validate_graph_query(symbol_name, max_depth)
    if error:
        return {"error": error}
    try:
        project_key = resolve_project_key()
        return code_graph_store.get_symbol_call_tree(project_key, symbol_name.strip(), max_depth)
    except Exception as e:
        return {"error": f"Failed tracing symbol call tree: {str(e)}"}


def get_type_hierarchy(symbol_name: str, max_depth: int = 3) -> dict[str, Any]:
    """Returns bounded base types and derived implementations for a symbol."""
    error = _validate_graph_query(symbol_name, max_depth)
    if error:
        return {"error": error}
    try:
        project_key = resolve_project_key()
        return code_graph_store.get_type_hierarchy(project_key, symbol_name.strip(), max_depth)
    except Exception as e:
        return {"error": f"Failed querying type hierarchy: {str(e)}"}


def _lsp_diagnostic_context(report: Any) -> list[dict[str, Any]]:
    severity_map = {"ERROR": 1, "WARNING": 2, "INFO": 3, "HINT": 4}
    diagnostics: list[dict[str, Any]] = []
    for item in getattr(report, "diagnostics", [])[:100]:
        diagnostics.append(
            {
                "range": {
                    "start": {
                        "line": max(0, item.line - 1),
                        "character": max(0, item.column - 1),
                    },
                    "end": {
                        "line": max(0, item.end_line - 1),
                        "character": max(0, item.end_column - 1),
                    },
                },
                "severity": severity_map.get(getattr(item.severity, "value", ""), 1),
                "code": item.code or None,
                "source": item.source,
                "message": item.message,
            }
        )
    return diagnostics


def _lsp_position_offset(text: str, position: dict[str, Any]) -> int:
    line = position.get("line")
    character = position.get("character")
    if (
        isinstance(line, bool)
        or not isinstance(line, int)
        or line < 0
        or isinstance(character, bool)
        or not isinstance(character, int)
        or character < 0
    ):
        raise ValueError(
            "LSP edit positions must use non-negative integer line and character values."
        )

    lines = text.splitlines(keepends=True)
    if not lines:
        lines = [""]
    if text.endswith(("\n", "\r")):
        lines.append("")
    if line >= len(lines):
        raise ValueError("LSP edit line is outside the current document.")

    offset = sum(len(part) for part in lines[:line])
    line_text = lines[line].rstrip("\r\n")
    units = 0
    for index, value in enumerate(line_text):
        if units == character:
            return offset + index
        units += 2 if ord(value) > 0xFFFF else 1
        if units > character:
            raise ValueError("LSP edit position splits a UTF-16 surrogate pair.")
    if units == character:
        return offset + len(line_text)
    raise ValueError("LSP edit character is outside the current line.")


def _extract_code_action_edits(
    workspace_edit: Any, target_uri: str
) -> tuple[list[dict[str, Any]] | None, str | None]:
    if not isinstance(workspace_edit, dict):
        return None, "The selected action does not contain a workspace text edit."
    if "changes" in workspace_edit and "documentChanges" in workspace_edit:
        return None, "Workspace edits containing both changes formats are ambiguous."

    edits: list[dict[str, Any]] = []
    changes = workspace_edit.get("changes")
    if changes is not None:
        if not isinstance(changes, dict) or set(changes) != {target_uri}:
            return None, "Workspace edits may only modify the selected file."
        candidate_edits = changes.get(target_uri)
        if not isinstance(candidate_edits, list):
            return None, "Workspace edits contain an invalid text edit list."
        edits.extend(candidate_edits)
    else:
        document_changes = workspace_edit.get("documentChanges")
        if not isinstance(document_changes, list) or not document_changes:
            return None, "The selected action does not contain a workspace text edit."
        for change in document_changes:
            if not isinstance(change, dict) or "textDocument" not in change:
                return None, "Resource operations and command-only actions are not supported."
            text_document = change.get("textDocument")
            if not isinstance(text_document, dict) or text_document.get("uri") != target_uri:
                return None, "Workspace edits may only modify the selected file."
            candidate_edits = change.get("edits")
            if not isinstance(candidate_edits, list):
                return None, "Workspace edits contain an invalid text edit list."
            edits.extend(candidate_edits)

    if not edits or not all(isinstance(edit, dict) for edit in edits):
        return None, "The selected action contains no valid text edits."
    return edits, None


def _code_action_selection_id(server_name: str, action: dict[str, Any]) -> str:
    payload = json.dumps(
        {"server_name": server_name, "action": action},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def apply_code_action(
    path: str,
    line: int,
    character: int,
    server_name: str | None = None,
    action_index: int | None = None,
    action_title: str | None = None,
    selection_id: str | None = None,
) -> dict[str, Any]:
    """Lists LSP fixes or applies one explicitly selected, validated text edit."""
    try:
        if (
            isinstance(line, bool)
            or not isinstance(line, int)
            or line < 0
            or isinstance(character, bool)
            or not isinstance(character, int)
            or character < 0
        ):
            return {"error": "line and character must be non-negative integers."}
        selecting = any(
            value is not None for value in (server_name, action_index, action_title, selection_id)
        )
        if selecting and (
            not isinstance(server_name, str)
            or not server_name.strip()
            or isinstance(action_index, bool)
            or not isinstance(action_index, int)
            or action_index < 0
            or not isinstance(action_title, str)
            or not action_title.strip()
            or not isinstance(selection_id, str)
            or not selection_id.strip()
        ):
            return {
                "error": "server_name, action_index, action_title, and selection_id must be supplied together."
            }

        resolved = _resolve_tool_path(path)
        if not resolved.is_file():
            return {"error": f"Path '{path}' is not an existing file."}
        sandbox = get_active_sandbox()
        if sandbox is not None:
            try:
                resolved.relative_to(sandbox.resolve())
            except ValueError:
                return {"error": "Access denied: code actions must stay inside the active sandbox."}

        from .lsp import get_lsp_manager

        manager = get_lsp_manager()
        workspace_root = Path(manager.workspace_root).resolve()
        try:
            resolved.relative_to(workspace_root)
        except ValueError:
            return {"error": "Code actions must stay inside the active LSP workspace."}

        if selecting:
            write_error = _check_write_allowed(resolved)
            if write_error:
                return {"error": write_error}
            _enforce_write_scope(path, resolved)

        with resolved.open("r", encoding="utf-8", newline="") as source_file:
            original = source_file.read()
        report = manager.check_file_sync(resolved, original, timeout_ms=50.0)
        uri = resolved.as_uri()
        position = {"line": line, "character": character}
        server_results = manager.request_semantic_sync(
            "textDocument/codeAction",
            uri,
            position,
            document_text=original,
            diagnostics=_lsp_diagnostic_context(report),
            timeout_sec=0.05,
        )

        actions_by_server: dict[str, list[dict[str, Any]]] = {}
        available: list[dict[str, Any]] = []
        if isinstance(server_results, dict):
            for current_server, server_actions in sorted(server_results.items()):
                if not isinstance(current_server, str) or not isinstance(server_actions, list):
                    continue
                valid_actions = [item for item in server_actions if isinstance(item, dict)]
                actions_by_server[current_server] = valid_actions
                for index, action in enumerate(valid_actions):
                    available.append(
                        {
                            "server_name": current_server,
                            "action_index": index,
                            "title": action.get("title")
                            if isinstance(action.get("title"), str)
                            else "(untitled action)",
                            "kind": action.get("kind")
                            if isinstance(action.get("kind"), str)
                            else None,
                            "selection_id": _code_action_selection_id(current_server, action),
                        }
                    )

        if not selecting:
            return {
                "status": "available" if available else "no_actions",
                "path": str(resolved),
                "position": position,
                "actions": available,
            }

        action_list = actions_by_server.get(server_name or "", [])
        if action_index is None or action_index >= len(action_list):
            return {
                "error": "The selected LSP action is no longer available; no source was changed."
            }
        selected = action_list[action_index]
        if selected.get("title") != action_title:
            return {"error": "The selected LSP action changed; no source was changed."}
        if _code_action_selection_id(server_name or "", selected) != selection_id:
            return {"error": "The selected LSP action changed; no source was changed."}
        if selected.get("command"):
            return {
                "error": "Command-only or command-bearing LSP actions are not supported; no source was changed."
            }

        text_edits, edit_error = _extract_code_action_edits(selected.get("edit"), uri)
        if edit_error:
            return {"error": f"{edit_error} No source was changed."}
        assert text_edits is not None

        resolved_edits: list[tuple[int, int, str]] = []
        for text_edit in text_edits:
            edit_range = text_edit.get("range")
            new_text = text_edit.get("newText")
            if (
                not isinstance(edit_range, dict)
                or not isinstance(new_text, str)
                or not isinstance(edit_range.get("start"), dict)
                or not isinstance(edit_range.get("end"), dict)
            ):
                return {"error": "The selected workspace edit is malformed; no source was changed."}
            start = _lsp_position_offset(original, edit_range["start"])
            end = _lsp_position_offset(original, edit_range["end"])
            if start > end:
                return {
                    "error": "The selected workspace edit has a reversed range; no source was changed."
                }
            resolved_edits.append((start, end, new_text))

        ascending_edits = sorted(resolved_edits)
        if any(
            current[0] < previous[1] or current[0] == previous[0]
            for previous, current in zip(ascending_edits, ascending_edits[1:], strict=False)
        ):
            return {
                "error": "Overlapping workspace edits are not supported; no source was changed."
            }
        updated = original
        for start, end, new_text in reversed(ascending_edits):
            updated = updated[:start] + new_text + updated[end:]

        with resolved.open("r", encoding="utf-8", newline="") as source_file:
            if source_file.read() != original:
                return {
                    "error": "The selected file changed while resolving the action; no source was changed."
                }
        file_mode = stat.S_IMODE(resolved.stat().st_mode)
        descriptor, temp_name = tempfile.mkstemp(prefix=f".{resolved.name}.", dir=resolved.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as temp_file:
                temp_file.write(updated)
                temp_file.flush()
                os.fsync(temp_file.fileno())
            os.chmod(temp_name, file_mode)
            os.replace(temp_name, resolved)
        except Exception:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise

        if resolved.suffix in SUPPORTED_EXTENSIONS:
            try:
                code_graph_indexer.index_file(resolved)
            except Exception:
                pass
        return {
            "status": "success",
            "path": str(resolved),
            "action_title": action_title,
            "edits_applied": len(text_edits),
        }
    except ToolScopeViolationError as e:
        return e.to_dict()
    except Exception as e:
        return {"error": f"Failed applying LSP code action: {str(e)}"}


_SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"(Bearer\s+)[A-Za-z0-9\-._~+/]+=*", re.IGNORECASE),
        r"\1[REDACTED_TOKEN]",
    ),
    (
        re.compile(r"\bya29\.[A-Za-z0-9\-_]+"),
        "[REDACTED_GCLOUD_TOKEN]",
    ),
    (
        re.compile(
            r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9 ]*PRIVATE KEY-----"
        ),
        "[REDACTED_PRIVATE_KEY]",
    ),
    (
        re.compile(r'("private_key"\s*:\s*)"[^"]+"'),
        r'\1"[REDACTED_PRIVATE_KEY]"',
    ),
    (
        re.compile(
            r"\b(sk-[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AIza[0-9A-Za-z\-_]{35}|AKIA[0-9A-Z]{16})\b"
        ),
        "[REDACTED_SECRET]",
    ),
    (
        re.compile(
            r"((?:api[_-]?key|access[_-]?token|secret[_-]?key|client[_-]?secret|password)\s*[=:]\s*['\"]?)([A-Za-z0-9\-._~+/]{8,})(['\"]?)",
            re.IGNORECASE,
        ),
        r"\1[REDACTED_SECRET]\3",
    ),
]


def _redact_secrets(text: str) -> str:
    """Redacts sensitive credentials, tokens, and private keys from command output."""
    if not text:
        return text
    redacted = text
    for pattern, replacement in _SECRET_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def _window_command_output(
    text: str,
    max_budget: int = 10000,
    head_chars: int = 2000,
    tail_chars: int = 8000,
) -> tuple[str, bool]:
    """Windows command output preserving head and tail when exceeding max_budget."""
    if len(text) <= max_budget:
        return text, False
    omitted_text = text[head_chars:-tail_chars]
    omitted_bytes = len(omitted_text.encode("utf-8", errors="replace"))
    windowed = (
        text[:head_chars]
        + f"\n[... Output truncated: {omitted_bytes} bytes omitted ...]\n"
        + text[-tail_chars:]
    )
    return windowed, True


def _build_safe_env() -> dict[str, str]:
    """Constructs a minimal environment that does not forward host credentials."""
    return sandbox_environment(home="/tmp")


@lru_cache(maxsize=1)
def _bwrap_available() -> bool:
    """Checks whether bubblewrap can create the namespaces required by this policy."""
    if sys.platform != "linux":
        return False
    bwrap_path = shutil.which("bwrap")
    shell_path = shutil.which("bash")
    if not bwrap_path or not shell_path:
        return False
    try:
        subprocess.run(
            build_bwrap_probe_argv(bwrap_path=bwrap_path, shell_path=shell_path),
            check=True,
            capture_output=True,
            env=_build_safe_env(),
            shell=False,
            timeout=5,
            text=True,
        )
        return True
    except (OSError, subprocess.SubprocessError, SandboxUnavailableError):
        return False


def _find_workspace_root(path: Path) -> Path:
    """Returns the nearest project root without widening a non-project cwd."""
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return candidate
    return path


def _prepare_shell_execution(
    command: str,
    *,
    work_dir: Path,
    workspace: Path,
    network_access: bool,
    allow_reduced_isolation: bool,
    sanitized_git_config: Path,
) -> tuple[list[str], dict[str, str], str, str | None]:
    bwrap_path = shutil.which("bwrap")
    shell_path = shutil.which("bash")
    if bwrap_path and shell_path and _bwrap_available():
        return (
            build_bwrap_argv(
                command,
                bwrap_path=bwrap_path,
                shell_path=shell_path,
                workspace=workspace,
                cwd=work_dir,
                network_access=network_access,
                sanitized_git_config=sanitized_git_config,
            ),
            _build_safe_env(),
            "os",
            None,
        )

    if not allow_reduced_isolation:
        raise SandboxUnavailableError(
            "OS isolation is unavailable; command execution is blocked by default."
        )
    if not network_access:
        raise SandboxUnavailableError(
            "The reduced-isolation fallback cannot enforce network isolation; "
            "network_access must be explicitly approved to use it."
        )
    if not shell_path:
        raise SandboxUnavailableError("A POSIX shell is unavailable for reduced-isolation mode.")
    return (
        reduced_shell_argv(shell_path=shell_path, command=command),
        sandbox_environment(home="/tmp"),
        "reduced",
        "Reduced isolation: this fallback does not provide OS process isolation or filesystem isolation.",
    )


def _command_workspace(work_dir: Path, active_sandbox: Path | None) -> Path:
    return active_sandbox.resolve() if active_sandbox else _find_workspace_root(work_dir.resolve())


def _new_sandbox_runtime_dir() -> Path:
    runtime_dir = Path(tempfile.mkdtemp(prefix="cli-agent-sandbox-", dir="/tmp"))
    sanitized_git_config = runtime_dir / "git-config"
    sanitized_git_config.write_text("", encoding="utf-8")
    sanitized_git_config.chmod(0o444)
    return runtime_dir


def _cleanup_sandbox_runtime_dir(runtime_dir: Path | None) -> None:
    if runtime_dir is not None:
        shutil.rmtree(runtime_dir, ignore_errors=True)


def _cleanup_background_sandbox(entry: BackgroundProcessEntry) -> None:
    _cleanup_sandbox_runtime_dir(entry.sandbox_temp_dir)
    entry.sandbox_temp_dir = None


def run_command(
    command: str,
    cwd: str | None = None,
    timeout_seconds: int = 60,
    network_access: bool = False,
    allow_reduced_isolation: bool = False,
) -> dict[str, Any]:
    """Executes a command in OS isolation, returning bounded and redacted output."""
    isolation = "unavailable"
    runtime_dir: Path | None = None
    try:
        active_sandbox = _active_sandbox_var.get()
        default_dir = active_sandbox or Path.cwd().resolve()
        work_dir = _resolve_tool_path(cwd) if cwd else default_dir
        if active_sandbox and not work_dir.is_relative_to(active_sandbox.resolve()):
            return {
                "error": f"Sandbox security violation: cwd '{work_dir}' is outside active sandbox '{active_sandbox}'.",
                "isolation": "unavailable",
            }
        workspace = _command_workspace(work_dir, active_sandbox)
        runtime_dir = _new_sandbox_runtime_dir()
        args, env, isolation, warning = _prepare_shell_execution(
            command,
            work_dir=work_dir,
            workspace=workspace,
            network_access=network_access,
            allow_reduced_isolation=allow_reduced_isolation,
            sanitized_git_config=runtime_dir / "git-config",
        )
        proc = subprocess.run(
            args,
            shell=False,
            cwd=work_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        raw_stdout = _redact_secrets(proc.stdout)
        raw_stderr = _redact_secrets(proc.stderr)

        stdout, stdout_trunc = _window_command_output(raw_stdout)
        stderr, stderr_trunc = _window_command_output(raw_stderr)

        result = {
            "exit_code": proc.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "truncated": stdout_trunc or stderr_trunc,
            "isolation": isolation,
        }
        if warning:
            result["warning"] = warning
        return result
    except SandboxUnavailableError as e:
        return {"error": str(e), "isolation": "unavailable"}
    except subprocess.TimeoutExpired:
        return {
            "error": f"Command timed out after {timeout_seconds} seconds.",
            "isolation": isolation,
        }
    except Exception as e:
        return {"error": f"Command execution failed: {str(e)}", "isolation": isolation}
    finally:
        _cleanup_sandbox_runtime_dir(runtime_dir)


def start_background_process(
    command: str,
    cwd: str | None = None,
    network_access: bool = False,
    allow_reduced_isolation: bool = False,
) -> dict[str, Any]:
    """Spawns a non-blocking background process (e.g., dev server, watcher) and streams output to a log file."""
    isolation = "unavailable"
    runtime_dir: Path | None = None
    preserve_runtime_dir = False
    try:
        active_sandbox = _active_sandbox_var.get()
        default_dir = active_sandbox or Path.cwd().resolve()
        work_dir = _resolve_tool_path(cwd) if cwd else default_dir
        if active_sandbox and not work_dir.is_relative_to(active_sandbox.resolve()):
            return {
                "error": f"Sandbox security violation: cwd '{work_dir}' is outside active sandbox '{active_sandbox}'.",
                "isolation": "unavailable",
            }
        workspace = _command_workspace(work_dir, active_sandbox)
        runtime_dir = _new_sandbox_runtime_dir()
        args, env, isolation, warning = _prepare_shell_execution(
            command,
            work_dir=work_dir,
            workspace=workspace,
            network_access=network_access,
            allow_reduced_isolation=allow_reduced_isolation,
            sanitized_git_config=runtime_dir / "git-config",
        )

        job_id = f"bg-{uuid.uuid4().hex[:8]}"
        log_dir = Path("/tmp/cli-agent/bg_logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{job_id}.log"

        log_file = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(
            args,
            shell=False,
            cwd=work_dir,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        entry = BackgroundProcessEntry(
            job_id=job_id,
            command=command,
            cwd=work_dir,
            process=proc,
            log_path=log_path,
            started_at=time.time(),
            sandbox_temp_dir=runtime_dir if isolation == "os" else None,
        )
        _background_processes[job_id] = entry
        preserve_runtime_dir = isolation == "os"
        result = {
            "status": "started",
            "job_id": job_id,
            "pid": proc.pid,
            "command": command,
            "cwd": str(work_dir),
            "log_path": str(log_path),
            "isolation": isolation,
        }
        if warning:
            result["warning"] = warning
        return result
    except SandboxUnavailableError as e:
        return {"error": str(e), "isolation": "unavailable"}
    except Exception as e:
        return {"error": f"Failed starting background process: {str(e)}", "isolation": isolation}
    finally:
        if not preserve_runtime_dir:
            _cleanup_sandbox_runtime_dir(runtime_dir)


def check_background_process(job_id: str, tail_lines: int = 100) -> dict[str, Any]:
    """Checks the status and recent stdout/stderr logs of a background process."""
    entry = _background_processes.get(job_id)
    if not entry:
        return {"error": f"Background job '{job_id}' not found."}

    ret_code = entry.process.poll()
    running = ret_code is None
    if not running:
        _cleanup_background_sandbox(entry)
    raw_logs = ""
    if entry.log_path.exists():
        try:
            lines = entry.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            raw_logs = "\n".join(lines[-max(1, tail_lines) :])
        except Exception:
            raw_logs = ""

    redacted_logs = _redact_secrets(raw_logs)
    windowed_logs, truncated = _window_command_output(redacted_logs)

    return {
        "job_id": job_id,
        "pid": entry.process.pid,
        "command": entry.command,
        "status": "running" if running else "exited",
        "exit_code": ret_code,
        "uptime_seconds": round(time.time() - entry.started_at, 2),
        "logs": windowed_logs,
        "truncated": truncated,
    }


def stop_background_process(job_id: str) -> dict[str, Any]:
    """Terminates a running background process group cleanly."""
    entry = _background_processes.get(job_id)
    if not entry:
        return {"error": f"Background job '{job_id}' not found."}

    ret_code = entry.process.poll()
    if ret_code is None:
        try:
            os.killpg(os.getpgid(entry.process.pid), signal.SIGTERM)
            try:
                entry.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(entry.process.pid), signal.SIGKILL)
                entry.process.wait(timeout=2)
        except ProcessLookupError:
            pass
        except Exception:
            entry.process.terminate()

    final_code = entry.process.poll()
    _cleanup_background_sandbox(entry)
    return {
        "status": "stopped",
        "job_id": job_id,
        "pid": entry.process.pid,
        "exit_code": final_code,
    }


async def delegate_task(
    role: str,
    objective: str,
    allowed_files: list[str] | None = None,
    context_files: list[str] | None = None,
) -> dict[str, Any]:
    """Delegates a discrete task to an isolated subagent worker (worker or qa)."""
    from .team import AgentRole, TaskAssignment, get_active_team

    team = get_active_team()
    if not team:
        return {"error": "Multi-agent team coordinator is not active in this session."}

    role_clean = role.lower().strip()
    target_role = AgentRole.WORKER if role_clean in ("worker", "coder", "dev") else AgentRole.QA
    assignment = TaskAssignment(
        task_id=f"subtask-{uuid.uuid4().hex[:6]}",
        role=target_role,
        objective=objective,
        allowed_files=allowed_files or [],
        context_files=context_files or [],
    )

    result = await team.spawn_worker(assignment, depth=1)
    return {
        "status": result.status,
        "task_id": result.task_id,
        "role": result.role.value,
        "summary": result.summary,
        "files_modified": result.files_modified,
        "error": result.error,
    }


# Tool Dispatcher
TOOL_REGISTRY: dict[str, Callable[..., Any]] = {
    "read_file": read_file,
    "list_directory": list_directory,
    "search_code": search_code,
    "lsp_workspace_diagnostics": lsp_workspace_diagnostics,
    "ast_grep_replace": ast_grep_replace,
    "hybrid_search_code": hybrid_search_code,
    "semantic_search_code": semantic_search_code,
    "get_symbol_call_tree": get_symbol_call_tree,
    "get_type_hierarchy": get_type_hierarchy,
    "apply_code_action": apply_code_action,
    "save_memory": save_memory,
    "search_memory": search_memory,
    "write_file": write_file,
    "edit_file": edit_file,
    "run_command": run_command,
    "start_background_process": start_background_process,
    "check_background_process": check_background_process,
    "stop_background_process": stop_background_process,
    "get_symbol_outline": get_symbol_outline,
    "find_symbol_definition": find_symbol_definition,
    "find_symbol_references": find_symbol_references,
    "analyze_symbol_impact": analyze_symbol_impact,
    "delegate_task": delegate_task,
}


def execute_tool(name: str, arguments: dict[str, Any]) -> Any:
    """Executes a tool by name with arguments."""
    if name.startswith("mcp__"):
        from .mcp import get_active_mcp_registry

        registry = get_active_mcp_registry()
        if not registry:
            return {"error": "No active MCP registry configured."}
        return registry.execute_tool(name, arguments)

    func = TOOL_REGISTRY.get(name)
    if not func:
        return {"error": f"Tool '{name}' is not registered."}
    try:
        return func(**arguments)
    except Exception as e:
        return {"error": f"Exception executing tool '{name}': {str(e)}"}


# Vertex AI Tool Declarations Schema
GEMINI_TOOLS_DECLARATIONS = [
    {
        "name": "read_file",
        "description": "Reads lines from a local file on the user's machine.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "path": {"type": "STRING", "description": "Relative or absolute path to the file."},
                "start_line": {
                    "type": "INTEGER",
                    "description": "1-indexed starting line number (default: 1).",
                },
                "end_line": {
                    "type": "INTEGER",
                    "description": "1-indexed ending line number (default: 300).",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "list_directory",
        "description": "Explores the files and folders inside a directory on the local machine.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "path": {
                    "type": "STRING",
                    "description": "Directory path to explore (default: current directory '.').",
                },
                "max_depth": {
                    "type": "INTEGER",
                    "description": "Depth of nested folders to inspect (default: 2).",
                },
            },
        },
    },
    {
        "name": "search_code",
        "description": "Searches for text or regex patterns across files in a local directory.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "pattern": {
                    "type": "STRING",
                    "description": "Text or regex pattern to search for.",
                },
                "path": {
                    "type": "STRING",
                    "description": "Directory where to search (default: '.').",
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "lsp_workspace_diagnostics",
        "description": "Collects bounded Python LSP and AST diagnostics without modifying files.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "path": {
                    "type": "STRING",
                    "description": "Python file or workspace directory to inspect.",
                },
                "max_files": {
                    "type": "INTEGER",
                    "description": "Maximum Python files to inspect (default: 250; maximum: 1000).",
                },
                "timeout_ms": {
                    "type": "INTEGER",
                    "description": "Per-file LSP diagnostic wait budget (default: 50 ms).",
                },
            },
        },
    },
    {
        "name": "ast_grep_replace",
        "description": "Previews a structural AST rewrite as a diff; it never modifies files.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "pattern": {"type": "STRING", "description": "ast-grep structural pattern."},
                "replacement": {
                    "type": "STRING",
                    "description": "Replacement pattern to preview.",
                },
                "language": {
                    "type": "STRING",
                    "description": "Language used to parse and match the pattern.",
                },
                "path": {
                    "type": "STRING",
                    "description": "File or directory to inspect (default: '.').",
                },
            },
            "required": ["pattern", "replacement", "language"],
        },
    },
    {
        "name": "hybrid_search_code",
        "description": "Performs hybrid code search combining sparse keyword matching (BM25/FTS5) and dense semantic vector embeddings via Reciprocal Rank Fusion (RRF). Ideal for conceptual, structural, and semantic code queries.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {
                    "type": "STRING",
                    "description": "Search query describing the feature, concept, or identifier to find in code.",
                },
                "limit": {
                    "type": "INTEGER",
                    "description": "Maximum number of code chunks to return (default: 5).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "semantic_search_code",
        "description": "Searches code by intent using local dense embeddings and sparse keyword retrieval; returns the best matching code chunks.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {
                    "type": "STRING",
                    "description": "Natural-language description of code behavior to find.",
                },
                "limit": {"type": "INTEGER", "description": "Maximum result count (default: 5)."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_symbol_call_tree",
        "description": "Traces bounded upstream callers and downstream callees for an indexed symbol across files.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "symbol_name": {
                    "type": "STRING",
                    "description": "Function, method, or symbol to trace.",
                },
                "max_depth": {
                    "type": "INTEGER",
                    "description": "Traversal depth from 1 to 5 (default: 3).",
                },
            },
            "required": ["symbol_name"],
        },
    },
    {
        "name": "get_type_hierarchy",
        "description": "Inspects indexed base classes and derived implementations for a type.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "symbol_name": {
                    "type": "STRING",
                    "description": "Class or type whose inheritance hierarchy to inspect.",
                },
                "max_depth": {
                    "type": "INTEGER",
                    "description": "Traversal depth from 1 to 5 (default: 3).",
                },
            },
            "required": ["symbol_name"],
        },
    },
    {
        "name": "apply_code_action",
        "description": "Lists LSP fixes at a zero-based file position. To apply one, call again with its server_name, action_index, exact action_title, and selection_id; only validated text edits to that file are applied, never server commands or multi-file/resource operations.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "path": {
                    "type": "STRING",
                    "description": "Workspace file to inspect and optionally update.",
                },
                "line": {
                    "type": "INTEGER",
                    "description": "Zero-based line containing the code action.",
                },
                "character": {
                    "type": "INTEGER",
                    "description": "Zero-based UTF-16 character position containing the code action.",
                },
                "server_name": {
                    "type": "STRING",
                    "description": "Server name returned by the listing call; required to apply.",
                },
                "action_index": {
                    "type": "INTEGER",
                    "description": "Action index returned by the listing call; required to apply.",
                },
                "action_title": {
                    "type": "STRING",
                    "description": "Exact action title returned by the listing call; required to apply.",
                },
                "selection_id": {
                    "type": "STRING",
                    "description": "Exact selection_id returned by the listing call; binds the action contents to the explicit choice.",
                },
            },
            "required": ["path", "line", "character"],
        },
    },
    {
        "name": "save_memory",
        "description": "Persists an important project decision, bug fix finding, convention, or note into persistent SQLite memory across sessions.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "title": {
                    "type": "STRING",
                    "description": "Brief searchable title (e.g. 'Database choice and reason').",
                },
                "content": {
                    "type": "STRING",
                    "description": "Detailed observation or finding to remember for future sessions.",
                },
                "memory_type": {
                    "type": "STRING",
                    "description": "Category: decision | discovery | bugfix | convention | preference",
                },
                "scope": {
                    "type": "STRING",
                    "description": "Scope: 'project' (default) for current repo, or 'global' for cross-project preferences.",
                },
            },
            "required": ["title", "content"],
        },
    },
    {
        "name": "search_memory",
        "description": "Queries SQLite persistent memory for past session notes, architecture decisions, and learnings in this workspace.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {
                    "type": "STRING",
                    "description": "Keywords or topic to search for in past memories.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "write_file",
        "description": "Writes full content to a file, creating parent directories if needed.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "path": {
                    "type": "STRING",
                    "description": "Relative or absolute path to the destination file.",
                },
                "content": {"type": "STRING", "description": "Full file content to write."},
                "overwrite": {
                    "type": "BOOLEAN",
                    "description": "Must be true to overwrite an existing file.",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replaces a single, exact, and unique occurrence of target text with replacement text in a file.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "path": {"type": "STRING", "description": "Path to the file to edit."},
                "target": {"type": "STRING", "description": "Exact text block to replace."},
                "replacement": {"type": "STRING", "description": "New text block to insert."},
            },
            "required": ["path", "target", "replacement"],
        },
    },
    {
        "name": "run_command",
        "description": "Runs a shell command in an OS sandbox by default. Network access and reduced-isolation fallback require explicit high-risk approval.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "command": {"type": "STRING", "description": "Bash command string to execute."},
                "cwd": {
                    "type": "STRING",
                    "description": "Working directory (default is current directory).",
                },
                "timeout_seconds": {
                    "type": "INTEGER",
                    "description": "Maximum seconds to wait (default: 60).",
                },
                "network_access": {
                    "type": "BOOLEAN",
                    "description": "Allow network access; disabled by default and requires high-risk approval.",
                },
                "allow_reduced_isolation": {
                    "type": "BOOLEAN",
                    "description": "Allow a non-OS-isolated fallback only when Bubblewrap is unavailable; requires high-risk approval and network_access=true.",
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "get_symbol_outline",
        "description": "Extracts a lightweight structural outline (classes, functions, methods, signatures, docstrings) of a source file using Tree-sitter without reading full line-by-line implementation bodies.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "file_path": {
                    "type": "STRING",
                    "description": "Path to the Python source file to outline.",
                },
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "find_symbol_definition",
        "description": "Finds the exact file path, line numbers, signature, and docstring of a symbol definition (class, function, method) across the project AST index.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "symbol_name": {
                    "type": "STRING",
                    "description": "Name of the class, function, or method to locate.",
                },
            },
            "required": ["symbol_name"],
        },
    },
    {
        "name": "find_symbol_references",
        "description": "Finds all references, invocations (calls), inheritances, or imports of a symbol across the project AST knowledge graph.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "symbol_name": {
                    "type": "STRING",
                    "description": "Target symbol name to trace callers and references for.",
                },
            },
            "required": ["symbol_name"],
        },
    },
    {
        "name": "analyze_symbol_impact",
        "description": "Analyzes structural impact of modifying a symbol across the codebase, returning definitions, incoming caller references, dependent files, and a risk rating (LOW, MEDIUM, HIGH).",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "symbol_name": {
                    "type": "STRING",
                    "description": "Target symbol name (class, function, or method) to evaluate impact for.",
                },
            },
            "required": ["symbol_name"],
        },
    },
    {
        "name": "delegate_task",
        "description": "Delegates a discrete subtask to an isolated Feature Worker or QA Verifier.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "role": {
                    "type": "STRING",
                    "description": "Target subagent role: 'worker' for code edits, or 'qa' for test verification.",
                },
                "objective": {
                    "type": "STRING",
                    "description": "Detailed task instructions and acceptance criteria for the subagent.",
                },
                "allowed_files": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"},
                    "description": "Approved files the worker is permitted to modify.",
                },
                "context_files": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"},
                    "description": "Read-only reference files for context.",
                },
            },
            "required": ["role", "objective"],
        },
    },
    {
        "name": "start_background_process",
        "description": "Starts a background process in an OS sandbox by default. Network access and reduced-isolation fallback require explicit high-risk approval.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "command": {
                    "type": "STRING",
                    "description": "Shell command to run in the background.",
                },
                "cwd": {"type": "STRING", "description": "Optional working directory."},
                "network_access": {
                    "type": "BOOLEAN",
                    "description": "Allow network access; disabled by default and requires high-risk approval.",
                },
                "allow_reduced_isolation": {
                    "type": "BOOLEAN",
                    "description": "Allow a non-OS-isolated fallback only when Bubblewrap is unavailable; requires high-risk approval and network_access=true.",
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "check_background_process",
        "description": "Checks status and recent stdout/stderr logs of a running or exited background process.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "job_id": {
                    "type": "STRING",
                    "description": "Background job ID returned by start_background_process.",
                },
                "tail_lines": {
                    "type": "INTEGER",
                    "description": "Number of recent log lines to fetch (default: 100).",
                },
            },
            "required": ["job_id"],
        },
    },
    {
        "name": "stop_background_process",
        "description": "Stops and cleans up a background process by its job_id.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "job_id": {"type": "STRING", "description": "Background job ID to terminate."},
            },
            "required": ["job_id"],
        },
    },
]
