"""Optional wrappers around fast native command-line tools.

The native programs are accelerators, never runtime requirements. Callers can
fall back to the established Python implementations when a binary is absent
or cannot complete the requested operation.
"""

from __future__ import annotations

import base64
import binascii
import difflib
import json
import shutil
import subprocess
import tempfile
import threading
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

IGNORED_DIRECTORY_NAMES = {".git", ".venv", "node_modules", "__pycache__", ".next", "dist", "build"}


@dataclass(frozen=True)
class ToolCapability:
    """Cached availability and executable paths for optional accelerators."""

    has_ripgrep: bool = False
    has_fd: bool = False
    has_ast_grep: bool = False
    has_delta: bool = False
    has_difftastic: bool = False
    ripgrep_path: str | None = None
    fd_path: str | None = None
    ast_grep_path: str | None = None
    delta_path: str | None = None
    difftastic_path: str | None = None


def _is_ast_grep(path: str) -> bool:
    """Avoid mistaking the unrelated util-linux `sg` executable for ast-grep."""
    try:
        result = subprocess.run(
            [path, "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return "ast-grep" in f"{result.stdout}\n{result.stderr}".lower()


@lru_cache(maxsize=1)
def detect_tool_capabilities() -> ToolCapability:
    """Probe native binaries once per process and retain their resolved paths."""
    ripgrep_path = shutil.which("rg")
    fd_path = shutil.which("fd") or shutil.which("fdfind")
    ast_grep_path = shutil.which("ast-grep")
    if ast_grep_path is None:
        sg_path = shutil.which("sg")
        if sg_path and _is_ast_grep(sg_path):
            ast_grep_path = sg_path
    delta_path = shutil.which("delta")
    difftastic_path = shutil.which("difft") or shutil.which("difftastic")
    return ToolCapability(
        has_ripgrep=ripgrep_path is not None,
        has_fd=fd_path is not None,
        has_ast_grep=ast_grep_path is not None,
        has_delta=delta_path is not None,
        has_difftastic=difftastic_path is not None,
        ripgrep_path=ripgrep_path,
        fd_path=fd_path,
        ast_grep_path=ast_grep_path,
        delta_path=delta_path,
        difftastic_path=difftastic_path,
    )


def get_installation_recommendations() -> list[str]:
    """Return package-manager suggestions without installing or changing anything."""
    managers = [
        ("brew", "brew install ripgrep fd ast-grep git-delta difftastic"),
        ("apt-get", "sudo apt install ripgrep fd-find; cargo install ast-grep"),
        ("pacman", "sudo pacman -S ripgrep fd ast-grep git-delta difftastic"),
        (
            "nix-env",
            "Install ripgrep, fd, ast-grep, git-delta, and difftastic from nixpkgs.",
        ),
        ("cargo", "cargo install ast-grep"),
    ]
    return [f"{manager}: {command}" for manager, command in managers if shutil.which(manager)]


def _json_string(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    text_value = value.get("text")
    if isinstance(text_value, str):
        return text_value
    encoded = value.get("bytes")
    if isinstance(encoded, str):
        try:
            return base64.b64decode(encoded).decode("utf-8", errors="replace")
        except (ValueError, binascii.Error):
            return ""
    return ""


def _relative_label(raw_path: str, base_path: Path) -> str:
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = base_path / candidate
    try:
        return candidate.resolve().relative_to(base_path.resolve()).as_posix()
    except (OSError, ValueError):
        return candidate.as_posix()


def fast_search_code(
    pattern: str,
    path: str = ".",
    file_pattern: str = "*",
    limit: int = 50,
) -> dict[str, Any] | None:
    """Search with ripgrep's JSON stream; return None when Python fallback is needed."""
    executable = detect_tool_capabilities().ripgrep_path
    if executable is None:
        return None

    base_path = Path(path).expanduser().resolve()
    args = [
        executable,
        "--json",
        "--context",
        "1",
        "--max-count",
        str(max(1, limit)),
        "--ignore-case",
        "--engine",
        "pcre2",
        "--multiline",
    ]
    if file_pattern != "*":
        args.extend(["--glob", file_pattern])
    args.extend(["--", pattern, str(base_path)])

    process: subprocess.Popen[str] | None = None
    timer: threading.Timer | None = None
    timed_out = False

    def stop_on_timeout() -> None:
        nonlocal timed_out
        timed_out = True
        if process is not None:
            try:
                process.kill()
            except OSError:
                pass

    try:
        process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        timer = threading.Timer(15.0, stop_on_timeout)
        timer.daemon = True
        timer.start()
        matches: list[str] = []
        contexts: list[str] = []
        context_seen: set[str] = set()
        malformed = False
        stopped_at_limit = False

        if process.stdout is None:
            timer.cancel()
            return None

        for raw_line in process.stdout:
            try:
                event = json.loads(raw_line)
            except (json.JSONDecodeError, TypeError):
                malformed = True
                break
            event_type = event.get("type")
            if event_type not in {"match", "context"}:
                continue
            data = event.get("data", {})
            relative_path = _relative_label(_json_string(data.get("path")), base_path)
            line_text = _json_string(data.get("lines")).strip()
            if not relative_path or not line_text:
                continue
            line_number = data.get("line_number") or 1
            rendered = f"{relative_path}:{line_number}: {line_text}"
            if event_type == "match":
                matches.append(rendered)
                if len(matches) >= max(1, limit):
                    stopped_at_limit = True
                    try:
                        process.terminate()
                    except OSError:
                        pass
                    break
            elif rendered not in context_seen:
                contexts.append(rendered)
                context_seen.add(rendered)

        if malformed:
            try:
                process.kill()
            except OSError:
                pass
        return_code = process.wait()
        timer.cancel()
        if timed_out or malformed or (return_code not in {0, 1} and not stopped_at_limit):
            return None
        return {
            "pattern": pattern,
            "total_matches": len(matches),
            "matches": matches or ["No matches found."],
            "context": contexts,
        }
    except (OSError, subprocess.SubprocessError, ValueError):
        if timer is not None:
            timer.cancel()
        if process is not None:
            try:
                process.kill()
                process.wait(timeout=1)
            except (OSError, subprocess.SubprocessError):
                pass
        return None


def fast_list_directory(path: str = ".", max_depth: int = 2) -> dict[str, Any] | None:
    """Use fd to construct a sorted tree; return None for the Python fallback."""
    executable = detect_tool_capabilities().fd_path
    if executable is None or max_depth < 0:
        return None

    base_path = Path(path).expanduser().resolve()
    args = [
        executable,
        "--hidden",
        "--exclude",
        ".git",
        "--max-depth",
        str(max_depth + 1),
        ".",
        str(base_path),
        "--print0",
        "--color",
        "never",
    ]
    for ignored_name in sorted(IGNORED_DIRECTORY_NAMES - {".git"}):
        args.extend(["--exclude", ignored_name])

    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="surrogateescape",
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode not in {0, 1}:
        return None

    files_by_parent: dict[Path, list[Path]] = defaultdict(list)
    directories_by_parent: dict[Path, list[Path]] = defaultdict(list)
    for raw_entry in result.stdout.split("\0"):
        if not raw_entry:
            continue
        candidate = Path(raw_entry)
        if candidate.is_absolute():
            try:
                candidate = candidate.relative_to(base_path)
            except ValueError:
                continue
        if not candidate.parts or any(part in IGNORED_DIRECTORY_NAMES for part in candidate.parts):
            continue
        absolute = base_path / candidate
        is_directory = absolute.is_dir()
        max_entry_depth = max_depth if is_directory else max_depth + 1
        if len(candidate.parts) > max_entry_depth:
            continue
        if is_directory:
            directories_by_parent[candidate.parent].append(candidate)
        else:
            files_by_parent[candidate.parent].append(candidate)

        parent = candidate.parent
        while parent != Path(".") and parent.parts:
            if parent not in directories_by_parent[parent.parent]:
                directories_by_parent[parent.parent].append(parent)
            parent = parent.parent

    entries: list[str] = []
    truncated = False

    def append_entry(value: str) -> bool:
        nonlocal truncated
        if len(entries) >= 200:
            truncated = True
            return False
        entries.append(value)
        return True

    def render(directory: Path, depth: int) -> bool:
        for file_path in sorted(files_by_parent.get(directory, []), key=lambda item: item.name):
            if not append_entry(f"{'  ' * depth}📄 {file_path.name}"):
                return False
        if depth >= max_depth:
            return True
        children = set(directories_by_parent.get(directory, []))
        for child in sorted(children, key=lambda item: item.name):
            if not append_entry(f"{'  ' * depth}📁 {child.name}/"):
                return False
            if not render(child, depth + 1):
                return False
        return True

    render(Path("."), 0)
    if truncated:
        entries.append("... [truncated list of files]")
    return {"path": str(base_path), "tree": "\n".join(entries)}


def ast_grep_replace(
    pattern: str,
    replacement: str,
    language: str,
    path: str = ".",
) -> dict[str, Any]:
    """Return ast-grep's rewrite diff for review without applying it to any file."""
    target = Path(path).expanduser().resolve()
    if not target.exists():
        return {"error": f"Path '{path}' does not exist."}
    executable = detect_tool_capabilities().ast_grep_path
    if executable is None:
        return {"error": "ast-grep is not installed; structural rewrite preview is unavailable."}

    args = [
        executable,
        "run",
        "--pattern",
        pattern,
        "--rewrite",
        replacement,
        "--lang",
        language,
        "--color",
        "never",
        str(target),
    ]
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"error": f"Failed to run ast-grep: {error}"}
    if result.returncode != 0:
        return {"error": result.stderr.strip() or f"ast-grep exited with code {result.returncode}."}
    return {
        "status": "preview",
        "path": str(target),
        "diff": result.stdout if result.stdout.strip() else "No structural matches found.",
        "applied": False,
    }


def capability_summary(capabilities: ToolCapability | None = None) -> dict[str, bool]:
    """Expose native-tool availability as a stable diagnostic mapping."""
    available = capabilities or detect_tool_capabilities()
    return {
        "ripgrep": available.has_ripgrep,
        "fd": available.has_fd,
        "ast-grep": available.has_ast_grep,
        "delta": available.has_delta,
        "difftastic": available.has_difftastic,
    }


def format_terminal_diff(path: str, before: str, after: str) -> str:
    """Format an edit diff with optional native terminal tools, falling back to unified text."""
    if before == after:
        return ""

    file_path = Path(path)
    filename = file_path.name or "file"
    plain_diff = "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
        )
    )
    capabilities = detect_tool_capabilities()

    if capabilities.difftastic_path:
        suffix = file_path.suffix or ".txt"
        try:
            with tempfile.TemporaryDirectory(prefix="cli-agent-diff-") as temporary_dir:
                before_path = Path(temporary_dir) / f"before{suffix}"
                after_path = Path(temporary_dir) / f"after{suffix}"
                before_path.write_text(before, encoding="utf-8")
                after_path.write_text(after, encoding="utf-8")
                result = subprocess.run(
                    [
                        capabilities.difftastic_path,
                        "--color",
                        "always",
                        "--display",
                        "inline",
                        str(before_path),
                        str(after_path),
                    ],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=15,
                    check=False,
                )
            if result.returncode in {0, 1} and result.stdout.strip():
                return result.stdout.replace(str(before_path), "<before>").replace(
                    str(after_path), str(file_path)
                )
        except (OSError, subprocess.SubprocessError, ValueError):
            pass

    if capabilities.delta_path and plain_diff:
        try:
            result = subprocess.run(
                [capabilities.delta_path, "--paging=never"],
                input=plain_diff,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout
        except (OSError, subprocess.SubprocessError, ValueError):
            pass

    return plain_diff
