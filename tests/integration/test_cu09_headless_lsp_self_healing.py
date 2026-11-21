"""
Integration test for CU-09: Real-Time Headless LSP Diagnostic Injection & Zero-Test Self-Correction Loop
(Wave 5 Milestone W5.3).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cli_agent.tools import edit_file, write_file


@pytest.mark.integration
class TestCU09HeadlessLSPSelfHealing:
    def test_lsp_diagnostic_injection_and_inline_healing(self, tmp_path: Path) -> None:
        calc_path = tmp_path / "calculator.py"
        path_str = str(calc_path)

        # Turn 1: Worker writes python file containing syntax/type defects
        broken_code = (
            "def calculate_total(prices: list[float]) -> float:\n"
            "    total = 0.0\n"
            "    for p in prices:\n"
            "        total += p\n"
            "    return undefined_vat_multiplier * total\n"
        )
        t1_resp = write_file(path_str, broken_code, overwrite=True)

        # Assert Turn 1 LSP diagnostic injection
        assert t1_resp["status"] == "success"
        assert t1_resp["lsp_status"] == "diagnostics_found"
        assert t1_resp["lsp_errors"] >= 1
        assert "lsp_hint" in t1_resp
        assert len(t1_resp["lsp_diagnostics"]) >= 1

        first_diag = t1_resp["lsp_diagnostics"][0]
        assert first_diag["line"] >= 1
        assert first_diag["column"] >= 1
        assert first_diag["severity"] == "ERROR"
        assert "undefined_vat_multiplier" in first_diag["message"] or first_diag["code"] in (
            "F821",
            "undefined-name",
            "SyntaxError",
        )

        # Turn 2: Worker repairs the defect via edit_file based directly on LSP diagnostic metadata
        fixed_line = "    vat_multiplier = 1.21\n    return vat_multiplier * total\n"
        t2_resp = edit_file(
            path_str,
            target="    return undefined_vat_multiplier * total\n",
            replacement=fixed_line,
        )

        # Assert Turn 2 returns clean LSP status with 0 errors
        assert t2_resp["status"] == "success"
        assert t2_resp["lsp_status"] == "clean"
        assert t2_resp["lsp_errors"] == 0
        assert t2_resp["lsp_warnings"] == 0
        assert "lsp_hint" not in t2_resp

        # Confirm content on disk is valid
        content_on_disk = calc_path.read_text(encoding="utf-8")
        assert "vat_multiplier = 1.21" in content_on_disk
        assert "undefined_vat_multiplier" not in content_on_disk
