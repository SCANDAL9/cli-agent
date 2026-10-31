"""
Unit tests for Wave 5 Phase 3: Headless LSP Semantic Diagnostic Bridge (src/cli_agent/lsp.py).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cli_agent.events import EventBus, EventTopic, set_event_bus
from cli_agent.lsp import (
    DiagnosticSeverity,
    HeadlessLSPClient,
    HeadlessLSPManager,
    LSPDiagnostic,
    LSPDiagnosticReport,
    _fast_ast_syntax_check,
    _find_executable,
    _map_lsp_severity,
)
from cli_agent.tools import edit_file, write_file


@pytest.mark.unit
class TestLSPDataModels:
    def test_map_lsp_severity(self) -> None:
        assert _map_lsp_severity(1) == DiagnosticSeverity.ERROR
        assert _map_lsp_severity(2) == DiagnosticSeverity.WARNING
        assert _map_lsp_severity(3) == DiagnosticSeverity.INFORMATION
        assert _map_lsp_severity(4) == DiagnosticSeverity.HINT
        assert _map_lsp_severity(None) == DiagnosticSeverity.ERROR
        assert _map_lsp_severity(99) == DiagnosticSeverity.ERROR

    def test_lsp_diagnostic_serialization_and_format(self) -> None:
        diag = LSPDiagnostic(
            file_path="src/agent.py",
            line=42,
            column=10,
            end_line=42,
            end_column=25,
            severity=DiagnosticSeverity.ERROR,
            message="Undefined variable 'foo'",
            source="ruff",
            code="F821",
        )
        d = diag.to_dict()
        assert d["line"] == 42
        assert d["column"] == 10
        assert d["severity"] == "ERROR"
        assert d["source"] == "ruff"
        assert d["code"] == "F821"

        compact = diag.format_compact()
        assert "Line 42:10 [ERROR] (ruff [F821]): Undefined variable 'foo'" == compact

    def test_lsp_diagnostic_report_properties(self) -> None:
        err = LSPDiagnostic(
            file_path="a.py",
            line=1,
            column=1,
            end_line=1,
            end_column=2,
            severity=DiagnosticSeverity.ERROR,
            message="Err",
        )
        warn = LSPDiagnostic(
            file_path="a.py",
            line=2,
            column=1,
            end_line=2,
            end_column=2,
            severity=DiagnosticSeverity.WARNING,
            message="Warn",
        )
        info = LSPDiagnostic(
            file_path="a.py",
            line=3,
            column=1,
            end_line=3,
            end_column=2,
            severity=DiagnosticSeverity.INFORMATION,
            message="Info",
        )

        rep = LSPDiagnosticReport(
            file_path="a.py",
            uri="file:///a.py",
            diagnostics=[err, warn, info],
            latency_ms=12.5,
        )
        assert rep.error_count == 1
        assert rep.warning_count == 1
        assert not rep.is_clean

        clean_rep = LSPDiagnosticReport(
            file_path="a.py",
            uri="file:///a.py",
            diagnostics=[info],
            latency_ms=5.0,
        )
        assert clean_rep.error_count == 0
        assert clean_rep.warning_count == 0
        assert clean_rep.is_clean


@pytest.mark.unit
class TestFastASTSyntaxCheckFallback:
    def test_clean_python_ast_returns_no_diagnostics(self, tmp_path: Path) -> None:
        p = tmp_path / "valid.py"
        code = "def add(x: int, y: int) -> int:\n    return x + y\n"
        diags = _fast_ast_syntax_check(p, code)
        assert diags == []

    def test_broken_python_ast_catches_syntax_error(self, tmp_path: Path) -> None:
        p = tmp_path / "broken.py"
        code = "def broken(:\n    return 42\n"
        diags = _fast_ast_syntax_check(p, code)
        assert len(diags) == 1
        d = diags[0]
        assert d.severity == DiagnosticSeverity.ERROR
        assert d.source == "python-ast"
        assert d.code == "SyntaxError"
        assert d.line == 1
        assert "SyntaxError" in d.message

    def test_non_python_file_skipped(self, tmp_path: Path) -> None:
        p = tmp_path / "data.json"
        code = "{ invalid json"
        assert _fast_ast_syntax_check(p, code) == []


@pytest.mark.unit
class TestHeadlessLSPClientAndManager:
    @pytest.mark.asyncio
    async def test_find_executable_discovers_ruff(self) -> None:
        found = _find_executable("ruff")
        assert found is not None
        assert "ruff" in found

    @pytest.mark.asyncio
    async def test_live_ruff_server_jsonrpc_diagnostics(self, tmp_path: Path) -> None:
        ruff_bin = _find_executable("ruff")
        if not ruff_bin:
            pytest.skip("Ruff binary not available in environment.")

        client = HeadlessLSPClient(
            server_name="ruff",
            command=[ruff_bin, "server", "--preview"],
            workspace_root=tmp_path,
        )
        started = await client.start()
        assert started is True
        assert client.is_ready is True

        try:
            test_file = tmp_path / "mod_test.py"
            uri = test_file.as_uri()

            # 1. Broken code with undefined name F821
            broken_code = "def run():\n    return undefined_var_xyz\n"
            await client.notify_document(uri, broken_code)
            diags = await client.wait_for_diagnostics(uri, timeout_sec=1.5)

            assert len(diags) >= 1
            codes = [d.code for d in diags]
            assert "F821" in codes or "undefined-name" in codes
            err_diag = next(d for d in diags if d.code in ("F821", "undefined-name"))
            assert err_diag.severity == DiagnosticSeverity.ERROR
            assert "undefined_var_xyz" in err_diag.message

            # 2. Update to clean code
            clean_code = "def run() -> int:\n    return 42\n"
            await client.notify_document(uri, clean_code)
            diags_clean = await client.wait_for_diagnostics(uri, timeout_sec=1.5)
            assert not any(d.code in ("F821", "undefined-name") for d in diags_clean)
        finally:
            await client.stop()
            assert client.is_ready is False

    def test_lsp_manager_sync_check_and_telemetry(self, tmp_path: Path) -> None:
        bus = EventBus()
        set_event_bus(bus)
        sub = bus.subscribe(EventTopic.SUBPROC_LOG)

        mgr = HeadlessLSPManager(workspace_root=tmp_path)
        try:
            broken_py = tmp_path / "broken_sync.py"
            code = "def bad_func():\n    return missing_ident\n"
            report = mgr.check_file_sync(broken_py, code, timeout_ms=500.0)

            assert report.file_path == str(broken_py.resolve())
            assert report.error_count >= 1
            assert any(d.code in ("F821", "undefined-name") for d in report.diagnostics)

            # Check telemetry event was published
            logs = sub.drain()
            assert any(
                "missing_ident" in env.payload.line for env in logs if hasattr(env.payload, "line")
            )  # type: ignore[union-attr]
        finally:
            mgr.stop()
            set_event_bus(None)


@pytest.mark.unit
class TestToolEnrichmentIntegration:
    def test_write_file_and_edit_file_enrich_lsp_diagnostics(self, tmp_path: Path) -> None:
        target_file = tmp_path / "sample.py"
        path_str = str(target_file)

        # 1. Write file with syntax error
        broken_syntax = "def invalid_syntax(:\n    pass\n"
        res_write = write_file(path_str, broken_syntax, overwrite=True)

        assert res_write["status"] == "success"
        assert res_write["lsp_status"] == "diagnostics_found"
        assert res_write["lsp_errors"] >= 1
        assert "lsp_hint" in res_write
        assert any(d["code"] == "SyntaxError" for d in res_write["lsp_diagnostics"])

        # 2. Fix via edit_file to valid code
        valid_code = "def valid_syntax() -> int:\n    return 100\n"
        res_edit = edit_file(path_str, target=broken_syntax, replacement=valid_code)

        assert res_edit["status"] == "success"
        assert res_edit["lsp_status"] == "clean"
        assert res_edit["lsp_errors"] == 0
        assert res_edit["lsp_warnings"] == 0
        assert "lsp_hint" not in res_edit

    def test_write_and_edit_non_python_file_does_not_invoke_lsp(self, tmp_path: Path) -> None:
        target_file = tmp_path / "notes.txt"
        path_str = str(target_file)
        res_write = write_file(path_str, "Hello World\n", overwrite=True)
        assert res_write["status"] == "success"
        assert "lsp_status" not in res_write

        res_edit = edit_file(path_str, "Hello", "Greetings")
        assert res_edit["status"] == "success"
        assert "lsp_status" not in res_edit


@pytest.mark.unit
class TestHeadlessLSPClientEdgeCases:
    @pytest.mark.asyncio
    async def test_find_executable_nonexistent_returns_none(self) -> None:
        assert _find_executable("completely_nonexistent_binary_xyz_123") is None

    @pytest.mark.asyncio
    async def test_client_start_invalid_binary_returns_false(self, tmp_path: Path) -> None:
        client = HeadlessLSPClient(
            server_name="invalid",
            command=["/nonexistent/bin/path_does_not_exist"],
            workspace_root=tmp_path,
        )
        started = await client.start()
        assert started is False
        assert client.is_ready is False
        await client.stop()  # Safe to call when not started

    @pytest.mark.asyncio
    async def test_client_notify_document_when_not_ready(self, tmp_path: Path) -> None:
        client = HeadlessLSPClient(
            server_name="test",
            command=["echo"],
            workspace_root=tmp_path,
        )
        # Should not raise
        await client.notify_document("file:///tmp/test.py", "x = 1")
        diags = await client.wait_for_diagnostics("file:///tmp/test.py", timeout_sec=0.01)
        assert diags == []
        await client.stop()

    def test_lsp_manager_lifecycle_and_active_server_names(self, tmp_path: Path) -> None:
        mgr = HeadlessLSPManager(workspace_root=tmp_path)
        try:
            assert isinstance(mgr.active_server_names, list)
        finally:
            mgr.stop()
            mgr.stop()  # Idempotent second stop
