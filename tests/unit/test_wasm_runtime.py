"""Functional tests for the embedded WASI plugin runtime."""

import subprocess
import sys
from pathlib import Path

import pytest
import wasmtime

from cli_agent.wasm_runtime import MAX_OUTPUT_BYTES, WasmCapabilities, WasmPluginRuntime

PRINT_WAT = r"""
(module
  (import "wasi_snapshot_preview1" "fd_write"
    (func $fd_write (param i32 i32 i32 i32) (result i32)))
  (memory (export "memory") 1)
  (data (i32.const 8) "ok\n")
  (func (export "_start")
    (i32.store (i32.const 0) (i32.const 8))
    (i32.store (i32.const 4) (i32.const 3))
    (call $fd_write (i32.const 1) (i32.const 0) (i32.const 1) (i32.const 20))
    drop))
"""

PATH_OPEN_WAT = r"""
(module
  (import "wasi_snapshot_preview1" "path_open"
    (func $path_open (param i32 i32 i32 i32 i32 i64 i64 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "proc_exit" (func $proc_exit (param i32)))
  (memory (export "memory") 1)
  (data (i32.const 8) "marker.txt")
  (func (export "_start") (local $error i32)
    (local.set $error
      (call $path_open
        (i32.const 3) (i32.const 0) (i32.const 8) (i32.const 10)
        (i32.const 0) (i64.const 2) (i64.const 0) (i32.const 0) (i32.const 64)))
    (call $proc_exit (local.get $error)))
)
"""

CREATE_FILE_WAT = r"""
(module
  (import "wasi_snapshot_preview1" "path_open"
    (func $path_open (param i32 i32 i32 i32 i32 i64 i64 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "proc_exit" (func $proc_exit (param i32)))
  (memory (export "memory") 1)
  (data (i32.const 8) "created.txt")
  (func (export "_start") (local $error i32)
    (local.set $error
      (call $path_open
        (i32.const 3) (i32.const 0) (i32.const 8) (i32.const 11)
        (i32.const 1) (i64.const 64) (i64.const 0) (i32.const 0) (i32.const 64)))
    (call $proc_exit (local.get $error)))
)
"""

ENV_COUNT_WAT = r"""
(module
  (import "wasi_snapshot_preview1" "environ_sizes_get"
    (func $env_sizes (param i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "proc_exit" (func $proc_exit (param i32)))
  (memory (export "memory") 1)
  (func (export "_start") (local $error i32)
    (local.set $error (call $env_sizes (i32.const 0) (i32.const 4)))
    (if (local.get $error) (then (call $proc_exit (i32.const 100))))
    (call $proc_exit (i32.load (i32.const 0))))
)
"""

INFINITE_LOOP_WAT = r"""
(module
  (memory (export "memory") 1)
  (func (export "_start") (loop $forever (br $forever)))
)
"""


def _write_module(root: Path, wat: str) -> Path:
    module_path = root / "plugin.wasm"
    module_path.write_bytes(wasmtime.wat2wasm(wat))
    return module_path


def test_runs_wasi_module_and_captures_bounded_output(tmp_path: Path) -> None:
    _write_module(tmp_path, PRINT_WAT)

    result = WasmPluginRuntime(tmp_path).run("plugin.wasm")

    assert result.status == "completed"
    assert result.exit_code == 0
    assert result.stdout == "ok\n"
    assert result.stderr == ""
    assert result.module_sha256
    assert result.truncated is False


def test_repeated_plugin_calls_exit_cleanly_at_process_shutdown(tmp_path: Path) -> None:
    _write_module(tmp_path, PRINT_WAT)
    runtime_path = Path(__file__).parents[2] / "src" / "cli_agent" / "wasm_runtime.py"
    script = r"""
import importlib.util
import sys
from pathlib import Path

runtime_path, workspace = map(Path, sys.argv[1:])
spec = importlib.util.spec_from_file_location("standalone_wasm_runtime", runtime_path)
runtime_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runtime_module
spec.loader.exec_module(runtime_module)
runtime = runtime_module.WasmPluginRuntime(workspace)
for _ in range(12):
    result = runtime.run("plugin.wasm")
    if result.status != "completed" or result.stdout.rstrip() != "ok":
        raise SystemExit(f"unexpected invocation result: {result!r}")
print("12 invocations completed")
"""

    completed = subprocess.run(
        [sys.executable, "-c", script, str(runtime_path), str(tmp_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "12 invocations completed"


def test_output_capture_is_bounded_for_stdout_and_stderr(tmp_path: Path) -> None:
    payload = "x" * (MAX_OUTPUT_BYTES + 1)
    wat = f"""
(module
  (import "wasi_snapshot_preview1" "fd_write"
    (func $fd_write (param i32 i32 i32 i32) (result i32)))
  (memory (export "memory") 2)
  (data (i32.const 8) "{payload}")
  (func $write (param $fd i32)
    (i32.store (i32.const 0) (i32.const 8))
    (i32.store (i32.const 4) (i32.const {len(payload)}))
    (call $fd_write (local.get $fd) (i32.const 0) (i32.const 1) (i32.const 65552))
    drop)
  (func (export "_start")
    (call $write (i32.const 1))
    (call $write (i32.const 2))))
"""
    _write_module(tmp_path, wat)

    result = WasmPluginRuntime(tmp_path).run("plugin.wasm")

    assert result.status == "completed"
    assert result.stdout == "x" * MAX_OUTPUT_BYTES
    assert result.stderr == "x" * MAX_OUTPUT_BYTES
    assert result.truncated is True


def test_wasi_filesystem_is_closed_by_default_and_read_grants_are_read_only(
    tmp_path: Path,
) -> None:
    (tmp_path / "marker.txt").write_text("marker", encoding="utf-8")
    _write_module(tmp_path, PATH_OPEN_WAT)
    runtime = WasmPluginRuntime(tmp_path)

    denied = runtime.run("plugin.wasm")
    readable = runtime.run(
        "plugin.wasm",
        capabilities=WasmCapabilities(read_dirs=(".",)),
    )

    assert denied.exit_code != 0
    assert readable.exit_code == 0


def test_read_only_preopen_rejects_file_creation_and_write_grant_allows_it(
    tmp_path: Path,
) -> None:
    _write_module(tmp_path, CREATE_FILE_WAT)
    runtime = WasmPluginRuntime(tmp_path)

    denied = runtime.run(
        "plugin.wasm",
        capabilities=WasmCapabilities(read_dirs=(".",)),
    )
    assert denied.exit_code != 0
    assert not (tmp_path / "created.txt").exists()

    allowed = runtime.run(
        "plugin.wasm",
        capabilities=WasmCapabilities(write_dirs=(".",)),
    )

    assert allowed.exit_code == 0
    assert (tmp_path / "created.txt").exists()


def test_plugin_gets_only_explicit_environment_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WAVE8_PLUGIN_MARKER", "present")
    _write_module(tmp_path, ENV_COUNT_WAT)
    runtime = WasmPluginRuntime(tmp_path)

    no_environment = runtime.run("plugin.wasm")
    one_variable = runtime.run(
        "plugin.wasm",
        capabilities=WasmCapabilities(environment_variables=("WAVE8_PLUGIN_MARKER",)),
    )

    assert no_environment.exit_code == 0
    assert one_variable.exit_code == 1


def test_network_capability_is_not_inherited_by_default_and_only_explicitly_added(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        "wasmtime._bindings.wasi_config_inherit_network",
        lambda _config: calls.append("inherit_network"),
    )
    _write_module(tmp_path, PRINT_WAT)
    runtime = WasmPluginRuntime(tmp_path)

    default = runtime.run("plugin.wasm")
    explicit = runtime.run(
        "plugin.wasm",
        capabilities=WasmCapabilities(network_access=True),
    )

    assert default.status == "completed"
    assert explicit.status == "completed"
    assert calls == ["inherit_network"]


def test_fuel_limit_stops_non_terminating_guest(tmp_path: Path) -> None:
    _write_module(tmp_path, INFINITE_LOOP_WAT)

    result = WasmPluginRuntime(tmp_path, fuel_limit=10_000).run("plugin.wasm")

    assert result.status == "failed"
    assert result.exit_code is None
    assert "fuel" in (result.error or "").lower()


def test_module_path_must_remain_inside_the_workspace(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.wasm"
    outside.write_bytes(wasmtime.wat2wasm(PRINT_WAT))

    result = WasmPluginRuntime(tmp_path).run("../outside.wasm")

    assert result.status == "failed"
    assert "outside the workspace" in (result.error or "")


def test_tool_policy_requires_high_risk_approval_for_each_capability_set() -> None:
    from cli_agent.security import RiskLevel, SecurityPolicy

    policy = SecurityPolicy(auto_approve=True)
    no_grants = {"module_path": "plugin.wasm"}
    read_grant = {"module_path": "plugin.wasm", "read_dirs": ["src"]}
    write_grant = {"module_path": "plugin.wasm", "write_dirs": ["src"]}

    assert policy.evaluate_risk("run_wasm_plugin", no_grants) == RiskLevel.MEDIUM
    assert policy.evaluate_risk("run_wasm_plugin", read_grant) == RiskLevel.HIGH
    assert policy.check_permission("run_wasm_plugin", read_grant)[0] is False
    policy.grant_session_permission("run_wasm_plugin", read_grant)
    assert policy.check_permission("run_wasm_plugin", read_grant)[0] is True
    assert policy.check_permission("run_wasm_plugin", write_grant)[0] is False
