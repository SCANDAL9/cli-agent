"""
Local system and memory tools for the Gemini agent.
Enables file reading, directory exploration, code searching, and persistent SQLite memories.
"""

import os
import re
import subprocess
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .code_graph import CodeGraphIndexer, CodeGraphStore
from .memory import MemoryStore, resolve_project_key

memory_store = MemoryStore()
code_graph_store = CodeGraphStore()
code_graph_indexer = CodeGraphIndexer(code_graph_store)

_active_sandbox_dir: Path | None = None


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
    global _active_sandbox_dir
    _active_sandbox_dir = Path(sandbox_dir).resolve() if sandbox_dir else None


def get_active_sandbox() -> Path | None:
    """Returns the current active sandbox directory, or None if in direct workspace mode."""
    return _active_sandbox_dir


def _resolve_tool_path(raw_path: str) -> Path:
    """Resolves a path, remapping relative or workspace paths into the active sandbox when enabled."""
    p = Path(raw_path).expanduser()
    if _active_sandbox_dir:
        if not p.is_absolute():
            return (_active_sandbox_dir / p).resolve()
        try:
            rel = p.relative_to(Path.cwd().resolve())
            return (_active_sandbox_dir / rel).resolve()
        except ValueError:
            return p.resolve()
    return p.resolve()


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
        if resolved.exists() and not overwrite:
            return {"error": f"File '{path}' already exists. Pass overwrite=True to replace it."}

        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")

        if resolved.suffix == ".py":
            try:
                code_graph_indexer.index_file(resolved)
            except Exception:
                pass

        return {
            "status": "success",
            "path": str(resolved),
            "bytes_written": len(content.encode("utf-8")),
            "lines": len(content.splitlines()),
        }
    except Exception as e:
        return {"error": f"Failed writing to '{path}': {str(e)}"}


def edit_file(path: str, target: str, replacement: str) -> dict[str, Any]:
    """Replaces a unique occurrence of target string with replacement, with whitespace-normalized fallback."""
    try:
        resolved = _resolve_tool_path(path)
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

        if resolved.suffix == ".py":
            try:
                code_graph_indexer.index_file(resolved)
            except Exception:
                pass

        return {
            "status": "success",
            "path": str(resolved),
            "replacements_made": 1,
        }
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


def run_command(command: str, cwd: str | None = None, timeout_seconds: int = 60) -> dict[str, Any]:
    """Executes a bash shell command with head-tail output windowing, secret redaction, and timeout protections."""
    try:
        default_dir = _active_sandbox_dir or Path.cwd().resolve()
        work_dir = Path(cwd).expanduser().resolve() if cwd else default_dir
        proc = subprocess.run(
            command,
            shell=True,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        raw_stdout = _redact_secrets(proc.stdout)
        raw_stderr = _redact_secrets(proc.stderr)

        stdout, stdout_trunc = _window_command_output(raw_stdout)
        stderr, stderr_trunc = _window_command_output(raw_stderr)

        return {
            "exit_code": proc.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "truncated": stdout_trunc or stderr_trunc,
        }
    except subprocess.TimeoutExpired:
        return {"error": f"Command timed out after {timeout_seconds} seconds."}
    except Exception as e:
        return {"error": f"Command execution failed: {str(e)}"}


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
    "hybrid_search_code": hybrid_search_code,
    "save_memory": save_memory,
    "search_memory": search_memory,
    "write_file": write_file,
    "edit_file": edit_file,
    "run_command": run_command,
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
        "description": "Runs a shell command (e.g. pytest, git status, linters) and returns stdout and stderr.",
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
]
