"""Tests for optional native command-line accelerators."""

import base64
import json
import subprocess
import threading
from pathlib import Path

import pytest

from cli_agent import fast_tools
from cli_agent.doctor import run_diagnostics
from cli_agent.memory import MemoryStore

ORIGINAL_TIMER = threading.Timer


@pytest.mark.unit
def test_detect_tool_capabilities_finds_supported_binary_names(monkeypatch):
    found = {
        "rg": "/usr/bin/rg",
        "fd": None,
        "fdfind": "/usr/bin/fdfind",
        "ast-grep": None,
        "sg": "/usr/local/bin/sg",
        "delta": "/usr/bin/delta",
        "difft": "/usr/bin/difft",
    }
    monkeypatch.setattr(fast_tools.shutil, "which", lambda name: found.get(name))

    def run(args, **kwargs):
        assert args == ["/usr/local/bin/sg", "--version"]
        return subprocess.CompletedProcess(args, 0, stdout="ast-grep 0.40.0\n", stderr="")

    monkeypatch.setattr(fast_tools.subprocess, "run", run)
    fast_tools.detect_tool_capabilities.cache_clear()

    capabilities = fast_tools.detect_tool_capabilities()

    assert capabilities.has_ripgrep is True
    assert capabilities.ripgrep_path == "/usr/bin/rg"
    assert capabilities.has_fd is True
    assert capabilities.fd_path == "/usr/bin/fdfind"
    assert capabilities.has_ast_grep is True
    assert capabilities.ast_grep_path == "/usr/local/bin/sg"
    assert capabilities.has_delta is True
    assert capabilities.has_difftastic is True


@pytest.mark.unit
def test_detect_tool_capabilities_is_cached(monkeypatch):
    calls = []
    monkeypatch.setattr(fast_tools.shutil, "which", lambda name: calls.append(name) or None)
    fast_tools.detect_tool_capabilities.cache_clear()

    first = fast_tools.detect_tool_capabilities()
    second = fast_tools.detect_tool_capabilities()

    assert first is second
    assert calls == [
        "rg",
        "fd",
        "fdfind",
        "ast-grep",
        "sg",
        "delta",
        "difft",
        "difftastic",
    ]


@pytest.mark.unit
def test_fast_search_code_streams_ripgrep_json_and_normalizes_context(monkeypatch, tmp_path: Path):
    source = tmp_path / "example.py"
    source.write_text("before()\nTARGET()\nafter()\n", encoding="utf-8")
    events = [
        {"type": "begin", "data": {"path": {"text": str(source)}}},
        {
            "type": "context",
            "data": {
                "path": {"text": str(source)},
                "lines": {"text": "before()\n"},
                "line_number": 1,
            },
        },
        {
            "type": "match",
            "data": {
                "path": {"text": str(source)},
                "lines": {"text": "TARGET()\n"},
                "line_number": 2,
            },
        },
        {
            "type": "context",
            "data": {
                "path": {"text": str(source)},
                "lines": {"text": "after()\n"},
                "line_number": 3,
            },
        },
        {"type": "end", "data": {"path": {"text": str(source)}}},
    ]

    class Process:
        def __init__(self, args, **kwargs):
            self.args = args
            self.stdout = iter(json.dumps(event) + "\n" for event in events)
            self.returncode = 0

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(
        fast_tools,
        "detect_tool_capabilities",
        lambda: fast_tools.ToolCapability(has_ripgrep=True, ripgrep_path="/usr/bin/rg"),
    )
    monkeypatch.setattr(fast_tools.subprocess, "Popen", Process)

    result = fast_tools.fast_search_code("TARGET", str(tmp_path))

    assert result == {
        "pattern": "TARGET",
        "total_matches": 1,
        "matches": ["example.py:2: TARGET()"],
        "context": ["example.py:1: before()", "example.py:3: after()"],
    }


@pytest.mark.unit
def test_fast_search_code_returns_none_when_ripgrep_is_missing(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        fast_tools,
        "detect_tool_capabilities",
        lambda: fast_tools.ToolCapability(),
    )

    assert fast_tools.fast_search_code("needle", str(tmp_path)) is None


@pytest.mark.unit
def test_fast_list_directory_builds_sorted_tree_from_fd(monkeypatch, tmp_path: Path):
    (tmp_path / "alpha.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / ".hidden").write_text("hidden", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("pass", encoding="utf-8")

    observed = {}

    def run(args, **kwargs):
        observed["args"] = args
        return subprocess.CompletedProcess(
            args, 0, stdout="src/main.py\0src\0alpha.txt\0.hidden\0", stderr=""
        )

    monkeypatch.setattr(
        fast_tools,
        "detect_tool_capabilities",
        lambda: fast_tools.ToolCapability(has_fd=True, fd_path="/usr/bin/fd"),
    )
    monkeypatch.setattr(fast_tools.subprocess, "run", run)

    result = fast_tools.fast_list_directory(str(tmp_path), max_depth=2)

    assert observed["args"][:7] == [
        "/usr/bin/fd",
        "--hidden",
        "--exclude",
        ".git",
        "--max-depth",
        "3",
        ".",
    ]
    assert result == {
        "path": str(tmp_path),
        "tree": "📄 .hidden\n📄 alpha.txt\n📁 src/\n  📄 main.py",
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    ("max_depth", "fd_depth", "expected_tree"),
    [
        (0, "1", "📄 root.txt"),
        (1, "2", "📄 root.txt\n📁 child/\n  📄 nested.txt"),
    ],
)
def test_fast_list_directory_matches_legacy_file_depth_boundaries(
    monkeypatch, tmp_path: Path, max_depth: int, fd_depth: str, expected_tree: str
):
    (tmp_path / "root.txt").write_text("root", encoding="utf-8")
    child = tmp_path / "child"
    child.mkdir()
    (child / "nested.txt").write_text("nested", encoding="utf-8")
    grandchild = child / "grandchild"
    grandchild.mkdir()
    (grandchild / "deep.txt").write_text("deep", encoding="utf-8")
    observed: dict[str, list[str]] = {}
    paths = [
        "root.txt",
        "child",
        "child/nested.txt",
        "child/grandchild",
        "child/grandchild/deep.txt",
    ]

    def run(args, **kwargs):
        observed["args"] = args
        return subprocess.CompletedProcess(args, 0, stdout="\0".join(paths) + "\0", stderr="")

    monkeypatch.setattr(
        fast_tools,
        "detect_tool_capabilities",
        lambda: fast_tools.ToolCapability(has_fd=True, fd_path="/usr/bin/fd"),
    )
    monkeypatch.setattr(fast_tools.subprocess, "run", run)

    result = fast_tools.fast_list_directory(str(tmp_path), max_depth=max_depth)

    assert observed["args"][5] == fd_depth
    assert result["tree"] == expected_tree


@pytest.mark.unit
def test_ast_grep_replace_is_a_non_mutating_preview(monkeypatch, tmp_path: Path):
    source = tmp_path / "sample.py"
    original = "foo(first, second)\n"
    source.write_text(original, encoding="utf-8")
    observed = {}
    diff = (
        "--- a/sample.py\n+++ b/sample.py\n@@ -1 +1 @@\n-foo(first, second)\n+bar(second, first)\n"
    )

    def run(args, **kwargs):
        observed["args"] = args
        return subprocess.CompletedProcess(args, 0, stdout=diff, stderr="")

    monkeypatch.setattr(
        fast_tools,
        "detect_tool_capabilities",
        lambda: fast_tools.ToolCapability(has_ast_grep=True, ast_grep_path="/usr/bin/ast-grep"),
    )
    monkeypatch.setattr(fast_tools.subprocess, "run", run)

    result = fast_tools.ast_grep_replace("foo($A, $B)", "bar($B, $A)", "python", str(source))

    assert result["status"] == "preview"
    assert result["diff"] == diff
    assert "--pattern" in observed["args"]
    assert "--rewrite" in observed["args"]
    assert "--lang" in observed["args"]
    assert "--update-all" not in observed["args"]
    assert source.read_text(encoding="utf-8") == original


@pytest.mark.unit
@pytest.mark.asyncio
async def test_doctor_reports_native_capabilities_and_install_recommendations(
    monkeypatch, tmp_path: Path
):
    class TokenProvider:
        async def get_token(self):
            return "valid-test-token"

    class PreflightEngine:
        def run_checks(self):
            from types import SimpleNamespace

            return SimpleNamespace(
                items=[], passed=True, account="test-account", project="test-project"
            )

    from cli_agent import preflight

    monkeypatch.setattr(preflight, "PreflightEngine", PreflightEngine)

    capabilities = fast_tools.ToolCapability(has_ripgrep=True, has_fd=True)
    monkeypatch.setattr(fast_tools, "detect_tool_capabilities", lambda: capabilities)
    monkeypatch.setattr(
        fast_tools,
        "get_installation_recommendations",
        lambda: ["brew: brew install ripgrep fd"],
    )
    memory_store = MemoryStore(db_path=tmp_path / "doctor-native-tools.db")

    report = await run_diagnostics(
        token_provider=TokenProvider(),
        memory_store=memory_store,
    )

    assert report["native_accelerators"]["ripgrep"] is True
    assert report["native_accelerators"]["fd"] is True
    assert report["native_accelerators"]["ast-grep"] is False
    assert report["native_tool_recommendations"] == ["brew: brew install ripgrep fd"]


@pytest.mark.unit
def test_format_terminal_diff_prefers_difftastic_and_cleans_temporary_files(monkeypatch):
    observed = {}

    def run(args, **kwargs):
        before = Path(args[-2])
        after = Path(args[-1])
        observed["before"] = before
        observed["after"] = after
        observed["contents"] = (
            before.read_text(encoding="utf-8"),
            after.read_text(encoding="utf-8"),
        )
        observed["args"] = args
        return subprocess.CompletedProcess(args, 1, stdout=f"semantic diff {after}\n", stderr="")

    monkeypatch.setattr(
        fast_tools,
        "detect_tool_capabilities",
        lambda: fast_tools.ToolCapability(has_difftastic=True, difftastic_path="/usr/bin/difft"),
    )
    monkeypatch.setattr(fast_tools.subprocess, "run", run)

    result = fast_tools.format_terminal_diff("module.py", "old = 1\n", "new = 2\n")

    assert result == "semantic diff module.py\n"
    assert "cli-agent-diff-" not in result
    assert observed["contents"] == ("old = 1\n", "new = 2\n")
    assert "--color" in observed["args"]
    assert observed["before"].suffix == ".py"
    assert not observed["before"].exists()
    assert not observed["after"].exists()


@pytest.mark.unit
def test_format_terminal_diff_uses_delta_then_plain_unified_fallback(monkeypatch):
    observed = {}

    def run(args, **kwargs):
        observed["args"] = args
        observed["input"] = kwargs["input"]
        return subprocess.CompletedProcess(args, 0, stdout="highlighted diff\n", stderr="")

    monkeypatch.setattr(
        fast_tools,
        "detect_tool_capabilities",
        lambda: fast_tools.ToolCapability(has_delta=True, delta_path="/usr/bin/delta"),
    )
    monkeypatch.setattr(fast_tools.subprocess, "run", run)

    result = fast_tools.format_terminal_diff("module.py", "old = 1\n", "new = 2\n")

    assert result == "highlighted diff\n"
    assert observed["args"][:2] == ["/usr/bin/delta", "--paging=never"]
    assert "--- a/module.py" in observed["input"]
    assert "+++ b/module.py" in observed["input"]


@pytest.mark.unit
def test_format_terminal_diff_returns_unified_diff_without_native_formatter(monkeypatch):
    monkeypatch.setattr(fast_tools, "detect_tool_capabilities", fast_tools.ToolCapability)

    result = fast_tools.format_terminal_diff("module.py", "old = 1\n", "new = 2\n")

    assert "--- a/module.py" in result
    assert "+++ b/module.py" in result
    assert "-old = 1" in result
    assert "+new = 2" in result


@pytest.mark.unit
def test_installation_recommendations_only_include_available_package_managers(monkeypatch):
    available = {"brew", "apt-get", "cargo"}
    monkeypatch.setattr(
        fast_tools.shutil, "which", lambda name: f"/usr/bin/{name}" if name in available else None
    )

    recommendations = fast_tools.get_installation_recommendations()

    assert [item.split(":", 1)[0] for item in recommendations] == ["brew", "apt-get", "cargo"]


@pytest.mark.unit
def test_json_and_path_helpers_handle_encoded_and_external_values(tmp_path: Path):
    encoded = base64.b64encode(b"binary path").decode("ascii")
    external_path = (tmp_path.parent / "outside.py").resolve()

    assert fast_tools._json_string(None) == ""
    assert fast_tools._json_string({"text": 4}) == ""
    assert fast_tools._json_string({"bytes": encoded}) == "binary path"
    assert fast_tools._json_string({"bytes": "A"}) == ""
    assert fast_tools._relative_label(str(external_path), tmp_path) == str(external_path)


@pytest.mark.unit
def test_ast_grep_preview_reports_unavailable_and_execution_errors(monkeypatch, tmp_path: Path):
    source = tmp_path / "example.py"
    source.write_text("value = 1\n", encoding="utf-8")

    assert (
        "does not exist"
        in fast_tools.ast_grep_replace(
            "value = $A", "value = 2", "python", str(tmp_path / "missing.py")
        )["error"]
    )

    monkeypatch.setattr(fast_tools, "detect_tool_capabilities", fast_tools.ToolCapability)
    assert (
        "not installed"
        in fast_tools.ast_grep_replace("value = $A", "value = 2", "python", str(source))["error"]
    )

    monkeypatch.setattr(
        fast_tools,
        "detect_tool_capabilities",
        lambda: fast_tools.ToolCapability(has_ast_grep=True, ast_grep_path="/usr/bin/sg"),
    )
    monkeypatch.setattr(
        fast_tools.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 2, stdout="", stderr="invalid pattern"
        ),
    )
    assert (
        fast_tools.ast_grep_replace("value = $A", "value = 2", "python", str(source))["error"]
        == "invalid pattern"
    )


@pytest.mark.unit
def test_fast_search_returns_fallback_signal_for_process_failures(monkeypatch, tmp_path: Path):
    class Process:
        def __init__(self, stdout, returncode):
            self.stdout = stdout
            self.returncode = returncode
            self.killed = False

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.killed = True

    monkeypatch.setattr(
        fast_tools,
        "detect_tool_capabilities",
        lambda: fast_tools.ToolCapability(has_ripgrep=True, ripgrep_path="/usr/bin/rg"),
    )
    malformed = Process(iter(["not json\n"]), 0)
    monkeypatch.setattr(fast_tools.subprocess, "Popen", lambda *args, **kwargs: malformed)
    assert fast_tools.fast_search_code("needle", str(tmp_path)) is None
    assert malformed.killed is True

    failed = Process(iter([]), 2)
    monkeypatch.setattr(fast_tools.subprocess, "Popen", lambda *args, **kwargs: failed)
    assert fast_tools.fast_search_code("needle", str(tmp_path)) is None

    monkeypatch.setattr(
        fast_tools.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("missing rg")),
    )
    assert fast_tools.fast_search_code("needle", str(tmp_path)) is None


@pytest.mark.unit
def test_fast_search_handles_missing_stdout_timeout_and_match_limit(monkeypatch, tmp_path: Path):
    class Process:
        def __init__(self, stdout, returncode=0):
            self.stdout = stdout
            self.returncode = returncode
            self.terminated = False
            self.killed = False

        def wait(self, timeout=None):
            return self.returncode

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

    monkeypatch.setattr(
        fast_tools,
        "detect_tool_capabilities",
        lambda: fast_tools.ToolCapability(has_ripgrep=True, ripgrep_path="/usr/bin/rg"),
    )
    missing_stdout = Process(None)
    monkeypatch.setattr(fast_tools.subprocess, "Popen", lambda *args, **kwargs: missing_stdout)
    assert fast_tools.fast_search_code("needle", str(tmp_path)) is None

    class ImmediateTimer:
        def __init__(self, interval, callback):
            self.callback = callback

        def start(self):
            self.callback()

        def cancel(self):
            pass

    timed_out = Process(iter([]), -9)
    monkeypatch.setattr(fast_tools.subprocess, "Popen", lambda *args, **kwargs: timed_out)
    monkeypatch.setattr(fast_tools.threading, "Timer", ImmediateTimer)
    assert fast_tools.fast_search_code("needle", str(tmp_path)) is None
    assert timed_out.killed is True

    event = {
        "type": "match",
        "data": {
            "path": {"text": str(tmp_path / "example.py")},
            "lines": {"text": "needle\n"},
            "line_number": 1,
        },
    }
    capped = Process(iter([json.dumps(event), json.dumps(event)]), -15)
    monkeypatch.setattr(fast_tools.subprocess, "Popen", lambda *args, **kwargs: capped)
    monkeypatch.setattr(fast_tools.threading, "Timer", ORIGINAL_TIMER)
    result = fast_tools.fast_search_code("needle", str(tmp_path), limit=1)
    assert result["matches"] == ["example.py:1: needle"]
    assert capped.terminated is True


@pytest.mark.unit
def test_fast_list_directory_falls_back_on_unavailable_or_failed_fd(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(fast_tools, "detect_tool_capabilities", fast_tools.ToolCapability)
    assert fast_tools.fast_list_directory(str(tmp_path)) is None
    assert fast_tools.fast_list_directory(str(tmp_path), max_depth=-1) is None

    monkeypatch.setattr(
        fast_tools,
        "detect_tool_capabilities",
        lambda: fast_tools.ToolCapability(has_fd=True, fd_path="/usr/bin/fd"),
    )
    monkeypatch.setattr(
        fast_tools.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 2, stdout="", stderr=""),
    )
    assert fast_tools.fast_list_directory(str(tmp_path)) is None

    monkeypatch.setattr(
        fast_tools.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("fd unavailable")),
    )
    assert fast_tools.fast_list_directory(str(tmp_path)) is None


@pytest.mark.unit
def test_fast_list_directory_truncates_sorted_result_and_respects_depth(
    monkeypatch, tmp_path: Path
):
    paths = [f"file-{index:03d}.txt" for index in range(205)]
    paths.extend(["nested/one.txt", ".git/config", ".venv/lib.py"])
    monkeypatch.setattr(
        fast_tools,
        "detect_tool_capabilities",
        lambda: fast_tools.ToolCapability(has_fd=True, fd_path="/usr/bin/fd"),
    )
    monkeypatch.setattr(
        fast_tools.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout="\0".join(paths) + "\0", stderr=""
        ),
    )

    result = fast_tools.fast_list_directory(str(tmp_path), max_depth=1)

    assert "file-000.txt" in result["tree"]
    assert "nested/" not in result["tree"]
    assert ".git" not in result["tree"]
    assert "... [truncated list of files]" in result["tree"]


@pytest.mark.unit
def test_terminal_diff_falls_back_to_delta_when_difftastic_fails(monkeypatch):
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        if args[0] == "/usr/bin/difft":
            return subprocess.CompletedProcess(args, 2, stdout="", stderr="bad parse")
        return subprocess.CompletedProcess(args, 0, stdout="delta diff\n", stderr="")

    monkeypatch.setattr(
        fast_tools,
        "detect_tool_capabilities",
        lambda: fast_tools.ToolCapability(
            has_delta=True,
            delta_path="/usr/bin/delta",
            has_difftastic=True,
            difftastic_path="/usr/bin/difft",
        ),
    )
    monkeypatch.setattr(fast_tools.subprocess, "run", run)

    assert fast_tools.format_terminal_diff("x.py", "one\n", "two\n") == "delta diff\n"
    assert len(calls) == 2


@pytest.mark.unit
def test_terminal_diff_falls_back_to_plain_text_after_formatter_errors(monkeypatch):
    monkeypatch.setattr(
        fast_tools,
        "detect_tool_capabilities",
        lambda: fast_tools.ToolCapability(has_delta=True, delta_path="/usr/bin/delta"),
    )
    monkeypatch.setattr(
        fast_tools.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired(args[0], 1)),
    )

    result = fast_tools.format_terminal_diff("x.py", "one\n", "two\n")

    assert "-one" in result
    assert "+two" in result


@pytest.mark.unit
def test_capability_probe_rejects_unrelated_sg_and_handles_probe_errors(monkeypatch):
    monkeypatch.setattr(
        fast_tools.shutil, "which", lambda name: "/usr/bin/sg" if name == "sg" else None
    )
    monkeypatch.setattr(
        fast_tools.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout="setgid version 1", stderr=""
        ),
    )
    fast_tools.detect_tool_capabilities.cache_clear()
    assert fast_tools.detect_tool_capabilities().has_ast_grep is False

    monkeypatch.setattr(
        fast_tools.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("sg not executable")),
    )
    assert fast_tools._is_ast_grep("/usr/bin/sg") is False
    fast_tools.detect_tool_capabilities.cache_clear()


@pytest.mark.unit
def test_fast_search_passes_file_glob_and_deduplicates_context(monkeypatch, tmp_path: Path):
    source = tmp_path / "example.py"
    context = {
        "type": "context",
        "data": {
            "path": {"text": str(source)},
            "lines": {"text": "before()\n"},
            "line_number": 1,
        },
    }
    match = {
        "type": "match",
        "data": {
            "path": {"text": str(source)},
            "lines": {"text": "needle()\n"},
            "line_number": 3,
        },
    }
    observed = {}

    class Process:
        stdout = iter(
            json.dumps(event) + "\n"
            for event in [
                context,
                context,
                {"type": "match", "data": {"lines": {"text": "\n"}}},
                match,
            ]
        )

        def wait(self, timeout=None):
            return 0

    def popen(args, **kwargs):
        observed["args"] = args
        return Process()

    monkeypatch.setattr(
        fast_tools,
        "detect_tool_capabilities",
        lambda: fast_tools.ToolCapability(has_ripgrep=True, ripgrep_path="/usr/bin/rg"),
    )
    monkeypatch.setattr(fast_tools.subprocess, "Popen", popen)

    result = fast_tools.fast_search_code("needle", str(tmp_path), file_pattern="*.py")

    assert "--glob" in observed["args"]
    assert "*.py" in observed["args"]
    assert result["matches"] == ["example.py:3: needle()"]
    assert result["context"] == ["example.py:1: before()"]


@pytest.mark.unit
def test_fast_list_directory_builds_missing_parents_and_skips_external_paths(
    monkeypatch, tmp_path: Path
):
    nested = tmp_path / "nested" / "deeper"
    nested.mkdir(parents=True)
    outside = (tmp_path.parent / "outside.py").resolve()
    monkeypatch.setattr(
        fast_tools,
        "detect_tool_capabilities",
        lambda: fast_tools.ToolCapability(has_fd=True, fd_path="/usr/bin/fd"),
    )
    monkeypatch.setattr(
        fast_tools.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout=f"{outside}\0nested/deeper\0", stderr=""
        ),
    )

    result = fast_tools.fast_list_directory(str(tmp_path), max_depth=2)

    assert result["tree"] == "📁 nested/\n  📁 deeper/"


@pytest.mark.unit
def test_fast_list_directory_truncates_inside_nested_directory(monkeypatch, tmp_path: Path):
    paths = [f"root-{index:03d}.txt" for index in range(101)]
    paths.extend(f"dir-{index:03d}/child.py" for index in range(50))
    monkeypatch.setattr(
        fast_tools,
        "detect_tool_capabilities",
        lambda: fast_tools.ToolCapability(has_fd=True, fd_path="/usr/bin/fd"),
    )
    monkeypatch.setattr(
        fast_tools.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout="\0".join(paths) + "\0", stderr=""
        ),
    )

    result = fast_tools.fast_list_directory(str(tmp_path), max_depth=2)

    assert "... [truncated list of files]" in result["tree"]
    assert len(result["tree"].splitlines()) == 201


@pytest.mark.unit
def test_terminal_diff_skips_identical_text_and_falls_back_after_delta_failure(monkeypatch):
    monkeypatch.setattr(
        fast_tools,
        "detect_tool_capabilities",
        lambda: fast_tools.ToolCapability(has_delta=True, delta_path="/usr/bin/delta"),
    )
    monkeypatch.setattr(
        fast_tools.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 2, stdout="", stderr="formatter failed"
        ),
    )

    assert fast_tools.format_terminal_diff("x.py", "same\n", "same\n") == ""
    fallback = fast_tools.format_terminal_diff("x.py", "one\n", "two\n")
    assert "-one" in fallback
    assert "+two" in fallback


@pytest.mark.unit
def test_capability_probe_prefers_ast_grep_binary(monkeypatch):
    observed = []

    def which(name):
        observed.append(name)
        return "/usr/bin/ast-grep" if name == "ast-grep" else None

    monkeypatch.setattr(fast_tools.shutil, "which", which)
    fast_tools.detect_tool_capabilities.cache_clear()

    capabilities = fast_tools.detect_tool_capabilities()

    assert capabilities.ast_grep_path == "/usr/bin/ast-grep"
    assert "sg" not in observed
    fast_tools.detect_tool_capabilities.cache_clear()


@pytest.mark.unit
def test_fast_search_survives_timeout_and_process_cleanup_errors(monkeypatch, tmp_path: Path):
    class ImmediateTimer:
        def __init__(self, interval, callback):
            self.callback = callback

        def start(self):
            self.callback()

        def cancel(self):
            pass

    class Process:
        def __init__(self, stdout, returncode=0):
            self.stdout = stdout
            self.returncode = returncode
            self.terminate_attempted = False

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            raise OSError("already stopped")

        def terminate(self):
            self.terminate_attempted = True
            raise OSError("already exited")

    monkeypatch.setattr(
        fast_tools,
        "detect_tool_capabilities",
        lambda: fast_tools.ToolCapability(has_ripgrep=True, ripgrep_path="/usr/bin/rg"),
    )
    monkeypatch.setattr(fast_tools.threading, "Timer", ImmediateTimer)
    timed_out = Process(iter([]), -9)
    monkeypatch.setattr(fast_tools.subprocess, "Popen", lambda *args, **kwargs: timed_out)
    assert fast_tools.fast_search_code("needle", str(tmp_path)) is None

    monkeypatch.setattr(fast_tools.threading, "Timer", ORIGINAL_TIMER)
    event = {
        "type": "match",
        "data": {
            "path": {"text": str(tmp_path / "example.py")},
            "lines": {"text": "needle\n"},
            "line_number": 1,
        },
    }
    capped = Process(iter([json.dumps(event)]), -15)
    monkeypatch.setattr(fast_tools.subprocess, "Popen", lambda *args, **kwargs: capped)
    result = fast_tools.fast_search_code("needle", str(tmp_path), limit=1)
    assert result["matches"] == ["example.py:1: needle"]
    assert capped.terminate_attempted is True

    malformed = Process(iter(["invalid json\n"]))
    monkeypatch.setattr(fast_tools.subprocess, "Popen", lambda *args, **kwargs: malformed)
    assert fast_tools.fast_search_code("needle", str(tmp_path)) is None


@pytest.mark.unit
def test_fast_search_cleans_up_when_wait_raises(monkeypatch, tmp_path: Path):
    timers = []

    class RecordingTimer:
        def __init__(self, interval, callback):
            self.callback = callback
            self.cancelled = False
            timers.append(self)

        def start(self):
            pass

        def cancel(self):
            self.cancelled = True

    class Process:
        stdout = iter([])
        calls = 0

        def wait(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired("rg", 15)
            return -9

        def kill(self):
            self.killed = True

    process = Process()
    monkeypatch.setattr(
        fast_tools,
        "detect_tool_capabilities",
        lambda: fast_tools.ToolCapability(has_ripgrep=True, ripgrep_path="/usr/bin/rg"),
    )
    monkeypatch.setattr(fast_tools.threading, "Timer", RecordingTimer)
    monkeypatch.setattr(fast_tools.subprocess, "Popen", lambda *args, **kwargs: process)

    assert fast_tools.fast_search_code("needle", str(tmp_path)) is None
    assert process.killed is True
    assert timers[0].cancelled is True


@pytest.mark.unit
def test_fast_list_directory_stops_when_the_next_root_directory_exceeds_limit(
    monkeypatch, tmp_path: Path
):
    paths = [f"file-{index:03d}.txt" for index in range(200)] + ["nested/file.py"]
    monkeypatch.setattr(
        fast_tools,
        "detect_tool_capabilities",
        lambda: fast_tools.ToolCapability(has_fd=True, fd_path="/usr/bin/fd"),
    )
    monkeypatch.setattr(
        fast_tools.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout="\0".join(paths) + "\0", stderr=""
        ),
    )

    result = fast_tools.fast_list_directory(str(tmp_path), max_depth=2)

    assert "nested/" not in result["tree"]
    assert "... [truncated list of files]" in result["tree"]


@pytest.mark.unit
def test_terminal_diff_falls_back_after_difftastic_timeout(monkeypatch):
    monkeypatch.setattr(
        fast_tools,
        "detect_tool_capabilities",
        lambda: fast_tools.ToolCapability(has_difftastic=True, difftastic_path="/usr/bin/difft"),
    )
    monkeypatch.setattr(
        fast_tools.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired(args[0], 1)),
    )

    result = fast_tools.format_terminal_diff("x.py", "one\n", "two\n")
    assert "-one" in result
    assert "+two" in result
