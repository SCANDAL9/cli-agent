"""Capability-scoped execution for untrusted WebAssembly plugins."""

from __future__ import annotations

import hashlib
import os
import re
import select
import threading
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MAX_MODULE_BYTES = 16 * 1024 * 1024
MAX_ARGUMENTS = 128
MAX_ARGUMENT_LENGTH = 4096
MAX_CAPABILITY_PATHS = 16
MAX_ENVIRONMENT_VARIABLES = 32
MAX_MEMORY_BYTES = 256 * 1024 * 1024
MAX_OUTPUT_BYTES = 64 * 1024
MAX_FUEL = 100_000_000
ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ENGINE_LOCK = threading.Lock()
_SHARED_ENGINE: Any | None = None


def _file_descriptor_directory() -> str:
    if os.name != "posix":
        raise RuntimeError("WASI output capture requires POSIX file-descriptor paths.")
    for directory in ("/proc/self/fd", "/dev/fd"):
        if os.path.isdir(directory):
            return directory
    raise RuntimeError("WASI output capture requires /proc/self/fd or /dev/fd.")


class _BoundedOutputCapture:
    """Drain a WASI pipe while retaining only the configured output limit."""

    def __init__(self, descriptor_directory: str) -> None:
        self.output = bytearray()
        self.truncated = False
        self._error: OSError | None = None
        self._finish_error: RuntimeError | None = None
        self._finished = False
        self._stop = threading.Event()
        self._read_fd, self._write_fd = os.pipe()
        try:
            os.set_blocking(self._read_fd, False)
            self.file_path = f"{descriptor_directory}/{self._write_fd}"
            self._reader = threading.Thread(target=self._drain, daemon=True)
            self._reader.start()
        except Exception:
            os.close(self._read_fd)
            os.close(self._write_fd)
            raise

    def _drain(self) -> None:
        try:
            while not self._stop.is_set():
                readable, _, _ = select.select([self._read_fd], [], [], 0.1)
                if not readable:
                    continue
                try:
                    chunk = os.read(self._read_fd, 4096)
                except BlockingIOError:
                    continue
                if not chunk:
                    break
                remaining = MAX_OUTPUT_BYTES - len(self.output)
                if remaining > 0:
                    self.output.extend(chunk[:remaining])
                if len(chunk) > remaining:
                    self.truncated = True
        except OSError as exc:
            self._error = exc
        finally:
            os.close(self._read_fd)

    def _close_writer(self) -> None:
        if self._write_fd is not None:
            os.close(self._write_fd)
            self._write_fd = None

    def finish(self) -> tuple[str, bool]:
        if self._finished:
            if self._finish_error is not None:
                raise self._finish_error
            return self.output.decode("utf-8", errors="replace"), self.truncated

        self._close_writer()
        self._reader.join(timeout=5)
        if self._reader.is_alive():
            self._stop.set()
            self._reader.join(timeout=1)
            self._finish_error = RuntimeError("WASI output pipe did not close cleanly.")
        elif self._error is not None:
            self._finish_error = RuntimeError(f"Unable to capture WASI output: {self._error}")
        self._finished = True
        if self._finish_error is not None:
            raise self._finish_error
        return self.output.decode("utf-8", errors="replace"), self.truncated

    def __enter__(self) -> _BoundedOutputCapture:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        try:
            self.finish()
        except Exception:
            if exc_type is None:
                raise
        return False


def _shared_engine(wasmtime: Any) -> Any:
    """Reuse one thread-safe Wasmtime engine to avoid per-call runtime teardown races."""
    global _SHARED_ENGINE
    with _ENGINE_LOCK:
        if _SHARED_ENGINE is None:
            config = wasmtime.Config()
            config.consume_fuel = True
            _SHARED_ENGINE = wasmtime.Engine(config)
        return _SHARED_ENGINE


@dataclass(frozen=True, slots=True)
class WasmCapabilities:
    """Explicit WASI capabilities; an empty value exposes no host files or env."""

    read_dirs: tuple[str, ...] = ()
    write_dirs: tuple[str, ...] = ()
    environment_variables: tuple[str, ...] = ()
    network_access: bool = False


@dataclass(frozen=True, slots=True)
class WasmExecutionResult:
    """Bounded result returned by a WASI plugin invocation."""

    status: str
    exit_code: int | None
    stdout: str
    stderr: str
    truncated: bool
    module_sha256: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class WasmPluginRuntime:
    """Runs WASI Preview 1 modules with bounded memory, fuel, and host access."""

    def __init__(
        self,
        workspace_root: Path | str,
        *,
        fuel_limit: int = 10_000_000,
        memory_limit_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        self.workspace_root = Path(workspace_root).expanduser().resolve(strict=True)
        if not self.workspace_root.is_dir():
            raise ValueError("The WASM workspace root must be a directory.")
        if not 1 <= fuel_limit <= MAX_FUEL:
            raise ValueError(f"fuel_limit must be between 1 and {MAX_FUEL}.")
        if not 1 <= memory_limit_bytes <= MAX_MEMORY_BYTES:
            raise ValueError(f"memory_limit_bytes must be between 1 and {MAX_MEMORY_BYTES}.")
        self.fuel_limit = fuel_limit
        self.memory_limit_bytes = memory_limit_bytes

    def run(
        self,
        module_path: str | Path,
        *,
        arguments: Sequence[str] = (),
        capabilities: WasmCapabilities | None = None,
    ) -> WasmExecutionResult:
        granted = capabilities or WasmCapabilities()
        try:
            self._validate_arguments(arguments)
            if not isinstance(granted.network_access, bool):
                raise ValueError("network_access must be a boolean.")
            module_file = self._resolve_module(module_path)
            module_bytes = module_file.read_bytes()
            if len(module_bytes) > MAX_MODULE_BYTES:
                raise ValueError(f"WASM module exceeds the {MAX_MODULE_BYTES}-byte limit.")
            module_digest = hashlib.sha256(module_bytes).hexdigest()
            read_dirs = self._resolve_directories(granted.read_dirs, "read_dirs")
            write_dirs = self._resolve_directories(granted.write_dirs, "write_dirs")
            environment = self._resolve_environment(granted.environment_variables)
            return self._execute(
                module_file,
                module_bytes,
                module_digest,
                arguments,
                read_dirs,
                write_dirs,
                environment,
                granted.network_access,
            )
        except Exception as exc:
            return WasmExecutionResult(
                status="failed",
                exit_code=None,
                stdout="",
                stderr="",
                truncated=False,
                error=str(exc)[:2048] or type(exc).__name__,
            )

    def _resolve_module(self, module_path: str | Path) -> Path:
        candidate = Path(module_path).expanduser()
        if not candidate.is_absolute():
            candidate = self.workspace_root / candidate
        resolved = candidate.resolve(strict=True)
        if not resolved.is_relative_to(self.workspace_root):
            raise ValueError("WASM module path is outside the workspace.")
        if not resolved.is_file():
            raise ValueError("WASM module path must be a regular file.")
        return resolved

    def _resolve_directories(self, paths: Sequence[str], field: str) -> tuple[Path, ...]:
        if not isinstance(paths, (list, tuple)):
            raise ValueError(f"{field} must be a list of workspace directories.")
        if len(paths) > MAX_CAPABILITY_PATHS:
            raise ValueError(f"{field} exceeds the {MAX_CAPABILITY_PATHS}-directory limit.")
        resolved_paths: list[Path] = []
        for path in paths:
            candidate = Path(path).expanduser()
            if not candidate.is_absolute():
                candidate = self.workspace_root / candidate
            resolved = candidate.resolve(strict=True)
            if not resolved.is_relative_to(self.workspace_root):
                raise ValueError(f"WASI {field} path is outside the workspace.")
            if not resolved.is_dir():
                raise ValueError(f"WASI {field} path must be a directory.")
            if resolved not in resolved_paths:
                resolved_paths.append(resolved)
        return tuple(resolved_paths)

    @staticmethod
    def _resolve_environment(names: Sequence[str]) -> tuple[tuple[str, str], ...]:
        if not isinstance(names, (list, tuple)):
            raise ValueError("environment_variables must be a list of variable names.")
        if len(names) > MAX_ENVIRONMENT_VARIABLES:
            raise ValueError(
                f"environment_variables exceeds the {MAX_ENVIRONMENT_VARIABLES}-variable limit."
            )
        selected: list[tuple[str, str]] = []
        for name in dict.fromkeys(names):
            if not isinstance(name, str) or not ENVIRONMENT_NAME.fullmatch(name):
                raise ValueError(f"Invalid environment variable name: {name!r}.")
            if name not in os.environ:
                raise ValueError(f"Requested environment variable is not set: {name}.")
            selected.append((name, os.environ[name]))
        return tuple(selected)

    @staticmethod
    def _validate_arguments(arguments: Sequence[str]) -> None:
        if not isinstance(arguments, (list, tuple)):
            raise ValueError("arguments must be a list of strings.")
        if len(arguments) > MAX_ARGUMENTS:
            raise ValueError(f"arguments exceeds the {MAX_ARGUMENTS}-argument limit.")
        if any(not isinstance(argument, str) or "\x00" in argument for argument in arguments):
            raise ValueError("WASI arguments must be strings without NUL characters.")
        if any(len(argument) > MAX_ARGUMENT_LENGTH for argument in arguments):
            raise ValueError(f"WASI arguments cannot exceed {MAX_ARGUMENT_LENGTH} characters.")

    def _execute(
        self,
        module_file: Path,
        module_bytes: bytes,
        module_digest: str,
        arguments: Sequence[str],
        read_dirs: tuple[Path, ...],
        write_dirs: tuple[Path, ...],
        environment: tuple[tuple[str, str], ...],
        network_access: bool,
    ) -> WasmExecutionResult:
        try:
            import wasmtime
            from wasmtime import _bindings as wasmtime_bindings
        except ImportError as exc:
            raise RuntimeError(
                "WASM plugin support requires the optional 'wasmtime' dependency."
            ) from exc

        engine = _shared_engine(wasmtime)
        descriptor_directory = _file_descriptor_directory()
        with (
            _BoundedOutputCapture(descriptor_directory) as stdout_capture,
            _BoundedOutputCapture(descriptor_directory) as stderr_capture,
        ):
            store = None
            linker = None
            module = None
            instance = None
            entry_point = None
            try:
                store = wasmtime.Store(engine)
                store.set_limits(
                    memory_size=self.memory_limit_bytes,
                    table_elements=10_000,
                    instances=1,
                    tables=1,
                    memories=1,
                )
                store.set_fuel(self.fuel_limit)

                wasi = wasmtime.WasiConfig()
                wasi.argv = [module_file.name, *arguments]
                if environment:
                    wasi.env = environment
                wasi.stdout_file = stdout_capture.file_path
                wasi.stderr_file = stderr_capture.file_path
                for index, path in enumerate(read_dirs):
                    wasi.preopen_dir(str(path), f"/grant/read/{index}", fs_mutable=False)
                for index, path in enumerate(write_dirs):
                    wasi.preopen_dir(str(path), f"/grant/write/{index}", fs_mutable=True)
                if network_access:
                    # Wasmtime exposes this stable C API, but wasmtime-py 51 does not wrap it.
                    inherit_network = getattr(
                        wasmtime_bindings, "wasi_config_inherit_network", None
                    )
                    if inherit_network is None:
                        raise RuntimeError("This Wasmtime build cannot grant WASI network access.")
                    inherit_network(wasi.ptr())

                store.set_wasi(wasi)
                linker = wasmtime.Linker(engine)
                linker.define_wasi()
                module = wasmtime.Module(engine, module_bytes)
                instance = linker.instantiate(store, module)
                entry_point = instance.exports(store).get("_start")
                if entry_point is None:
                    raise ValueError("WASI module must export the '_start' entry point.")
                try:
                    entry_point(store)
                    exit_code = 0
                except wasmtime.ExitTrap as exc:
                    exit_code = exc.code
            finally:
                del entry_point
                del instance
                try:
                    if linker is not None:
                        linker.close()
                finally:
                    try:
                        if module is not None:
                            module.close()
                    finally:
                        if store is not None:
                            store.close()

            stdout, stdout_truncated = stdout_capture.finish()
            stderr, stderr_truncated = stderr_capture.finish()
            return WasmExecutionResult(
                status="completed",
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                truncated=stdout_truncated or stderr_truncated,
                module_sha256=module_digest,
            )
