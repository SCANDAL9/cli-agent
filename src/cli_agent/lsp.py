"""
Headless Language Server Protocol (LSP) Semantic Diagnostic Bridge.

Provides a warm, asynchronous JSON-RPC 2.0 stdio LSP client bridge to pyright-langserver
and ruff server, delivering real-time type checking, linting, and AST-level diagnostics
inline inside write_file and edit_file tool invocations in < 50ms.
"""

from __future__ import annotations

import ast
import asyncio
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from .events import EventTopic, SubprocLogPayload, get_event_bus


class DiagnosticSeverity(StrEnum):
    """Normalized severity levels mapped from LSP DiagnosticSeverity (1..4)."""

    ERROR = "ERROR"  # LSP 1
    WARNING = "WARNING"  # LSP 2
    INFORMATION = "INFO"  # LSP 3
    HINT = "HINT"  # LSP 4


def _map_lsp_severity(sev: int | None) -> DiagnosticSeverity:
    if sev == 1:
        return DiagnosticSeverity.ERROR
    if sev == 2:
        return DiagnosticSeverity.WARNING
    if sev == 3:
        return DiagnosticSeverity.INFORMATION
    if sev == 4:
        return DiagnosticSeverity.HINT
    return DiagnosticSeverity.ERROR


@dataclass(frozen=True, slots=True)
class LSPDiagnostic:
    """Single normalized diagnostic item from an LSP server or AST fallback."""

    file_path: str
    line: int  # 1-indexed line number (converted from 0-indexed LSP Range)
    column: int  # 1-indexed column number
    end_line: int
    end_column: int
    severity: DiagnosticSeverity
    message: str
    source: str = "lsp"  # e.g., "pyright", "ruff", "python-ast"
    code: str = ""  # e.g., "reportArgumentType", "F821", "SyntaxError"

    def to_dict(self) -> dict[str, Any]:
        return {
            "line": self.line,
            "column": self.column,
            "end_line": self.end_line,
            "end_column": self.end_column,
            "severity": self.severity.value,
            "source": self.source,
            "code": self.code,
            "message": self.message,
        }

    def format_compact(self) -> str:
        code_str = f" [{self.code}]" if self.code else ""
        return (
            f"Line {self.line}:{self.column} [{self.severity.value}] "
            f"({self.source}{code_str}): {self.message}"
        )


@dataclass(frozen=True, slots=True)
class LSPDiagnosticReport:
    """Aggregated diagnostic result returned to write_file / edit_file."""

    file_path: str
    uri: str
    diagnostics: list[LSPDiagnostic] = field(default_factory=list)
    latency_ms: float = 0.0
    server_names: list[str] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(1 for d in self.diagnostics if d.severity == DiagnosticSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for d in self.diagnostics if d.severity == DiagnosticSeverity.WARNING)

    @property
    def is_clean(self) -> bool:
        return self.error_count == 0 and self.warning_count == 0


def _find_executable(name: str) -> str | None:
    """Finds an executable in PATH, virtualenv, or next to the current Python binary."""
    # 1. Virtualenv bin directory
    venv_bin = Path(sys.executable).parent / name
    if venv_bin.is_file() and os.access(venv_bin, os.X_OK):
        return str(venv_bin)

    # 2. Local project .venv
    local_venv = Path.cwd() / ".venv" / "bin" / name
    if local_venv.is_file() and os.access(local_venv, os.X_OK):
        return str(local_venv)

    # 3. System PATH
    found = shutil.which(name)
    if found:
        return found

    return None


class HeadlessLSPClient:
    """Manages a persistent, headless LSP server subprocess over stdio JSON-RPC 2.0."""

    def __init__(
        self,
        server_name: str,
        command: list[str],
        workspace_root: Path,
    ) -> None:
        self.server_name = server_name
        self.command = command
        self.workspace_root = workspace_root.resolve()
        self._proc: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._next_request_id = 1
        self._pending_requests: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._diagnostics_cache: dict[str, list[LSPDiagnostic]] = {}
        self._diagnostic_events: dict[str, asyncio.Event] = {}
        self._opened_docs: dict[str, int] = {}
        self._is_ready = False
        self._lock = asyncio.Lock()

    @property
    def is_ready(self) -> bool:
        return self._is_ready and self._proc is not None and self._proc.returncode is None

    async def start(self) -> bool:
        """Spawns the LSP subprocess and executes the LSP initialize handshake."""
        async with self._lock:
            if self.is_ready:
                return True

            try:
                self._proc = await asyncio.create_subprocess_exec(
                    *self.command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=str(self.workspace_root),
                )
            except Exception:
                self._proc = None
                return False

            self._reader_task = asyncio.create_task(
                self._stdout_reader_loop(), name=f"lsp-reader-{self.server_name}"
            )

            # Initialize handshake
            init_req = {
                "jsonrpc": "2.0",
                "id": self._get_request_id(),
                "method": "initialize",
                "params": {
                    "processId": os.getpid(),
                    "rootUri": self.workspace_root.as_uri(),
                    "capabilities": {
                        "textDocument": {
                            "synchronization": {
                                "didOpen": True,
                                "didChange": 1,  # Full sync
                                "didSave": True,
                            },
                            "publishDiagnostics": {
                                "relatedInformation": True,
                                "versionSupport": True,
                            },
                        },
                        "workspace": {
                            "configuration": True,
                        },
                    },
                },
            }

            fut = self._register_request(init_req["id"])  # type: ignore[arg-type]
            await self._send_message(init_req)

            try:
                resp = await asyncio.wait_for(fut, timeout=10.0)
                if "error" in resp:
                    await self.stop()
                    return False
            except Exception:
                await self.stop()
                return False

            # Send initialized notification
            await self._send_message({"jsonrpc": "2.0", "method": "initialized", "params": {}})
            self._is_ready = True
            return True

    def _get_request_id(self) -> int:
        req_id = self._next_request_id
        self._next_request_id += 1
        return req_id

    def _register_request(self, req_id: int) -> asyncio.Future[dict[str, Any]]:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending_requests[req_id] = fut
        return fut

    async def _send_message(self, message: dict[str, Any]) -> None:
        if not self._proc or not self._proc.stdin:
            return
        payload = json.dumps(message, ensure_ascii=False).encode("utf-8")
        header = f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii")
        try:
            self._proc.stdin.write(header + payload)
            await self._proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            self._is_ready = False

    async def _stdout_reader_loop(self) -> None:
        """Reads framed JSON-RPC 2.0 messages from LSP stdout."""
        if not self._proc or not self._proc.stdout:
            return
        stdout = self._proc.stdout

        while True:
            try:
                # 1. Read HTTP headers until empty \r\n line
                content_length: int | None = None
                while True:
                    line_bytes = await stdout.readline()
                    if not line_bytes:
                        # EOF
                        self._is_ready = False
                        return
                    line = line_bytes.decode("latin1").strip()
                    if not line:
                        break
                    if line.lower().startswith("content-length:"):
                        parts = line.split(":", 1)
                        if len(parts) == 2:
                            try:
                                content_length = int(parts[1].strip())
                            except ValueError:
                                pass

                if content_length is None:
                    continue

                # 2. Read exactly content_length bytes
                payload_bytes = await stdout.readexactly(content_length)
                payload_str = payload_bytes.decode("utf-8", errors="replace")
                data = json.loads(payload_str)
                self._dispatch_message(data)

            except asyncio.IncompleteReadError:
                break
            except asyncio.CancelledError:
                break
            except Exception:
                continue

    def _dispatch_message(self, msg: dict[str, Any]) -> None:
        # 1. Response to pending request
        msg_id = msg.get("id")
        if msg_id is not None and ("result" in msg or "error" in msg):
            if isinstance(msg_id, int) and msg_id in self._pending_requests:
                fut = self._pending_requests.pop(msg_id)
                if not fut.done():
                    fut.set_result(msg)
            return

        # 2. Server-to-client request (e.g., workspace/configuration)
        if msg_id is not None and "method" in msg:
            # Immediate dummy acknowledgment to prevent deadlock
            asyncio.create_task(self._send_message({"jsonrpc": "2.0", "id": msg_id, "result": []}))
            return

        # 3. Notification (e.g., textDocument/publishDiagnostics)
        method = msg.get("method")
        if method == "textDocument/publishDiagnostics":
            params = msg.get("params", {})
            uri = params.get("uri", "")
            raw_diags = params.get("diagnostics", [])
            normalized = self._normalize_diagnostics(uri, raw_diags)
            self._diagnostics_cache[uri] = normalized
            if uri in self._diagnostic_events:
                self._diagnostic_events[uri].set()

    def _normalize_diagnostics(
        self, uri: str, raw_diags: list[dict[str, Any]]
    ) -> list[LSPDiagnostic]:
        res: list[LSPDiagnostic] = []
        file_path = uri.removeprefix("file://")
        for rd in raw_diags:
            rng = rd.get("range", {})
            start = rng.get("start", {})
            end = rng.get("end", {})
            line = start.get("line", 0) + 1
            col = start.get("character", 0) + 1
            end_line = end.get("line", start.get("line", 0)) + 1
            end_col = end.get("character", start.get("character", 0)) + 1
            severity = _map_lsp_severity(rd.get("severity"))
            msg = rd.get("message", "")
            source = rd.get("source") or self.server_name
            data = rd.get("data")
            if isinstance(data, dict) and "code" in data and data["code"]:
                code = str(data["code"])
            else:
                code = str(rd.get("code") or "")

            res.append(
                LSPDiagnostic(
                    file_path=file_path,
                    line=line,
                    column=col,
                    end_line=end_line,
                    end_column=end_col,
                    severity=severity,
                    message=msg,
                    source=source,
                    code=code,
                )
            )
        return res

    async def notify_document(self, uri: str, content: str) -> None:
        """Sends didOpen or didChange for the document."""
        if not self.is_ready:
            return

        if uri not in self._diagnostic_events:
            self._diagnostic_events[uri] = asyncio.Event()
        self._diagnostic_events[uri].clear()

        if uri not in self._opened_docs:
            self._opened_docs[uri] = 1
            open_msg = {
                "jsonrpc": "2.0",
                "method": "textDocument/didOpen",
                "params": {
                    "textDocument": {
                        "uri": uri,
                        "languageId": "python",
                        "version": 1,
                        "text": content,
                    }
                },
            }
            await self._send_message(open_msg)
        else:
            self._opened_docs[uri] += 1
            v = self._opened_docs[uri]
            change_msg = {
                "jsonrpc": "2.0",
                "method": "textDocument/didChange",
                "params": {
                    "textDocument": {"uri": uri, "version": v},
                    "contentChanges": [{"text": content}],
                },
            }
            await self._send_message(change_msg)

    async def wait_for_diagnostics(self, uri: str, timeout_sec: float) -> list[LSPDiagnostic]:
        """Awaits push diagnostics for a URI up to timeout_sec."""
        if not self.is_ready:
            return list(self._diagnostics_cache.get(uri, []))

        event = self._diagnostic_events.get(uri)
        if event is None:
            return list(self._diagnostics_cache.get(uri, []))

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout_sec)
        except TimeoutError:
            pass
        return list(self._diagnostics_cache.get(uri, []))

    async def stop(self) -> None:
        """Gracefully shuts down the LSP server process."""
        self._is_ready = False
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()

        if self._proc and self._proc.returncode is None:
            try:
                # Send shutdown request
                req_id = self._get_request_id()
                fut = self._register_request(req_id)
                await self._send_message(
                    {"jsonrpc": "2.0", "id": req_id, "method": "shutdown", "params": None}
                )
                try:
                    await asyncio.wait_for(fut, timeout=2.0)
                except Exception:
                    pass

                # Send exit notification
                await self._send_message({"jsonrpc": "2.0", "method": "exit", "params": None})
                await asyncio.sleep(0.05)
            except Exception:
                pass

            if self._proc.returncode is None:
                try:
                    self._proc.terminate()
                    await asyncio.wait_for(self._proc.wait(), timeout=1.0)
                except Exception:
                    try:
                        self._proc.kill()
                    except Exception:
                        pass
        self._proc = None


def _fast_ast_syntax_check(file_path: Path, content: str) -> list[LSPDiagnostic]:
    """In-memory AST parser fallback to catch SyntaxError / IndentationError in <1ms."""
    if file_path.suffix != ".py":
        return []

    try:
        ast.parse(content, filename=str(file_path))
        return []
    except SyntaxError as e:
        lineno = e.lineno or 1
        col = e.offset or 1
        end_lineno = getattr(e, "end_lineno", None) or lineno
        end_col = getattr(e, "end_offset", None) or col
        msg = e.msg or str(e)
        return [
            LSPDiagnostic(
                file_path=str(file_path),
                line=lineno,
                column=col,
                end_line=end_lineno,
                end_column=end_col,
                severity=DiagnosticSeverity.ERROR,
                message=f"SyntaxError: {msg}",
                source="python-ast",
                code="SyntaxError",
            )
        ]
    except Exception as e:
        return [
            LSPDiagnostic(
                file_path=str(file_path),
                line=1,
                column=1,
                end_line=1,
                end_column=1,
                severity=DiagnosticSeverity.ERROR,
                message=f"ParserError: {str(e)}",
                source="python-ast",
                code="ParserError",
            )
        ]


class HeadlessLSPManager:
    """
    Coordinates warm headless LSP clients across workspaces and worktrees,
    managing an internal event loop thread for sub-50ms synchronous and
    asynchronous diagnostic queries.
    """

    def __init__(self, workspace_root: Path | None = None) -> None:
        self.workspace_root = (workspace_root or Path.cwd()).resolve()
        self._clients: list[HeadlessLSPClient] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._loop_ready = threading.Event()
        self._is_stopped = False
        self._lock = threading.Lock()
        self._ensure_loop_started()

    def _ensure_loop_started(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._loop_ready.clear()
            self._thread = threading.Thread(
                target=self._run_loop, name="cli-agent-lsp-thread", daemon=True
            )
            self._thread.start()
            self._loop_ready.wait(timeout=5.0)

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        self._loop_ready.set()
        loop.run_forever()

    async def _init_clients(self) -> None:
        if self._clients or self._is_stopped:
            return

        # 1. Discover ruff server
        ruff_bin = _find_executable("ruff")
        if ruff_bin:
            client = HeadlessLSPClient(
                server_name="ruff",
                command=[ruff_bin, "server", "--preview"],
                workspace_root=self.workspace_root,
            )
            started = await client.start()
            if started:
                self._clients.append(client)

        # 2. Discover pyright-langserver / basedpyright
        pyright_bin = _find_executable("pyright-langserver") or _find_executable(
            "basedpyright-langserver"
        )
        if pyright_bin:
            client = HeadlessLSPClient(
                server_name="pyright",
                command=[pyright_bin, "--stdio"],
                workspace_root=self.workspace_root,
            )
            started = await client.start()
            if started:
                self._clients.append(client)

    @property
    def active_server_names(self) -> list[str]:
        return [c.server_name for c in self._clients if c.is_ready]

    async def notify_and_collect_diagnostics(
        self,
        file_path: Path,
        content: str,
        timeout_ms: float = 50.0,
    ) -> LSPDiagnosticReport:
        """Asynchronously dispatches document updates and gathers diagnostics within timeout_ms."""
        t0 = time.perf_counter()
        resolved_path = file_path.resolve()
        uri = resolved_path.as_uri()

        await self._init_clients()

        # 1. If AST syntax error exists, capture it immediately
        ast_diags = _fast_ast_syntax_check(resolved_path, content)

        active_clients = [c for c in self._clients if c.is_ready]
        if not active_clients:
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return LSPDiagnosticReport(
                file_path=str(resolved_path),
                uri=uri,
                diagnostics=ast_diags,
                latency_ms=elapsed_ms,
                server_names=["python-ast"],
            )

        # 2. Dispatch notifications
        for client in active_clients:
            await client.notify_document(uri, content)

        # 3. Wait for push diagnostics
        budget_sec = max(0.005, timeout_ms / 1000.0)
        gathered: list[LSPDiagnostic] = list(ast_diags)
        server_names_reported: set[str] = {"python-ast"} if ast_diags else set()

        tasks = [client.wait_for_diagnostics(uri, budget_sec) for client in active_clients]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for client, res in zip(active_clients, results, strict=False):
            if isinstance(res, list):
                gathered.extend(res)
                server_names_reported.add(client.server_name)

        # Deduplicate diagnostics by (line, column, code, message)
        seen: set[tuple[int, int, str, str]] = set()
        deduped: list[LSPDiagnostic] = []
        for d in gathered:
            key = (d.line, d.column, d.code, d.message)
            if key not in seen:
                seen.add(key)
                deduped.append(d)

        # Sort diagnostics: errors first, then warnings, ordered by line and column
        def _sort_key(d: LSPDiagnostic) -> tuple[int, int, int]:
            sev_order = (
                0
                if d.severity == DiagnosticSeverity.ERROR
                else 1
                if d.severity == DiagnosticSeverity.WARNING
                else 2
            )
            return (sev_order, d.line, d.column)

        deduped.sort(key=_sort_key)

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        report = LSPDiagnosticReport(
            file_path=str(resolved_path),
            uri=uri,
            diagnostics=deduped,
            latency_ms=elapsed_ms,
            server_names=sorted(server_names_reported),
        )

        # Telemetry: Publish to EventBus if errors exist
        if report.error_count > 0:
            try:
                bus = get_event_bus()
                for d in report.diagnostics[:5]:
                    bus.publish_nowait(
                        EventTopic.SUBPROC_LOG,
                        SubprocLogPayload(
                            source="LSP",
                            line=d.format_compact(),
                            level="ERROR" if d.severity == DiagnosticSeverity.ERROR else "WARN",
                        ),
                    )
            except Exception:
                pass

        return report

    def check_file_sync(
        self,
        file_path: Path,
        content: str,
        timeout_ms: float = 50.0,
    ) -> LSPDiagnosticReport:
        """Synchronously checks diagnostics by dispatching to the background event loop."""
        self._ensure_loop_started()
        if not self._loop or not self._loop.is_running():
            # In-memory fallback if loop failed to start
            ast_diags = _fast_ast_syntax_check(file_path.resolve(), content)
            return LSPDiagnosticReport(
                file_path=str(file_path.resolve()),
                uri=file_path.resolve().as_uri(),
                diagnostics=ast_diags,
                latency_ms=0.5,
                server_names=["python-ast"],
            )

        budget_sec = max(0.1, (timeout_ms / 1000.0) + 0.1)
        fut = asyncio.run_coroutine_threadsafe(
            self.notify_and_collect_diagnostics(file_path, content, timeout_ms=timeout_ms),
            self._loop,
        )
        try:
            return fut.result(timeout=budget_sec)
        except Exception:
            ast_diags = _fast_ast_syntax_check(file_path.resolve(), content)
            return LSPDiagnosticReport(
                file_path=str(file_path.resolve()),
                uri=file_path.resolve().as_uri(),
                diagnostics=ast_diags,
                latency_ms=timeout_ms,
                server_names=["python-ast"],
            )

    def stop(self) -> None:
        """Terminates active LSP clients and stops the background thread."""
        with self._lock:
            if self._is_stopped:
                return
            self._is_stopped = True

            if self._loop and self._loop.is_running():

                async def _stop_all() -> None:
                    for client in self._clients:
                        await client.stop()
                    self._clients.clear()

                fut = asyncio.run_coroutine_threadsafe(_stop_all(), self._loop)
                try:
                    fut.result(timeout=3.0)
                except Exception:
                    pass

                self._loop.call_soon_threadsafe(self._loop.stop)

            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=2.0)
                self._thread = None
            self._loop = None


_GLOBAL_LSP_MANAGER: HeadlessLSPManager | None = None
_GLOBAL_LSP_LOCK = threading.Lock()


def get_lsp_manager(workspace_root: Path | None = None) -> HeadlessLSPManager:
    """Returns the process-wide singleton HeadlessLSPManager instance."""
    global _GLOBAL_LSP_MANAGER
    with _GLOBAL_LSP_LOCK:
        if _GLOBAL_LSP_MANAGER is None:
            _GLOBAL_LSP_MANAGER = HeadlessLSPManager(workspace_root)
        return _GLOBAL_LSP_MANAGER
