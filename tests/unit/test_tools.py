"""
Unit tests for local system and filesystem tools.
"""

from pathlib import Path

import pytest

from cli_agent.tools import (
    edit_file,
    execute_tool,
    list_directory,
    read_file,
    run_command,
    search_code,
    write_file,
)


@pytest.mark.unit
class TestToolsUnit:
    def test_write_and_read_file(self, tmp_path: Path):
        file_path = tmp_path / "subdir" / "test.txt"
        content = "Line 1\nLine 2\nLine 3\n"

        # 1. Write file
        write_res = write_file(str(file_path), content)
        assert write_res.get("status") == "success"
        assert write_res.get("lines") == 3
        assert file_path.exists()

        # 2. Prevent accidental overwrite
        conflict_res = write_file(str(file_path), "Different content", overwrite=False)
        assert "already exists" in conflict_res.get("error", "")

        # 3. Allow overwrite
        overwrite_res = write_file(str(file_path), "New Line 1\n", overwrite=True)
        assert overwrite_res.get("status") == "success"

        # 4. Read file
        read_res = read_file(str(file_path), start_line=1, end_line=10)
        assert "New Line 1" in read_res["content"]

    def test_read_file_nonexistent(self, tmp_path: Path):
        res = read_file(str(tmp_path / "does_not_exist.py"))
        assert "does not exist" in res.get("error", "")

    def test_edit_file_exact_match(self, tmp_path: Path):
        file_path = tmp_path / "code.py"
        file_path.write_text("def hello():\n    return 'world'\n", encoding="utf-8")

        # Successful edit
        res = edit_file(str(file_path), target="'world'", replacement="'antigravity'")
        assert res.get("status") == "success"
        assert res.get("replacements_made") == 1
        assert "return 'antigravity'" in file_path.read_text(encoding="utf-8")

    def test_edit_file_target_not_found(self, tmp_path: Path):
        file_path = tmp_path / "code.py"
        file_path.write_text("x = 1\n", encoding="utf-8")

        res = edit_file(str(file_path), target="y = 2", replacement="y = 3")
        assert "Target substring was not found" in res.get("error", "")

    def test_edit_file_multiple_occurrences_fails_for_safety(self, tmp_path: Path):
        file_path = tmp_path / "code.py"
        file_path.write_text("item = 1\nitem = 2\n", encoding="utf-8")

        res = edit_file(str(file_path), target="item", replacement="val")
        assert "occurs 2 times" in res.get("error", "")
        # File must remain untouched
        assert file_path.read_text(encoding="utf-8") == "item = 1\nitem = 2\n"

    def test_edit_file_whitespace_and_crlf_fallback(self, tmp_path: Path):
        file_path = tmp_path / "indented.py"
        file_path.write_bytes(b"class Engine:\r\n    def compute(self):\r\n        return 10\r\n")

        # 1. CRLF vs LF difference
        res_crlf = edit_file(
            str(file_path),
            target="    def compute(self):\n        return 10",
            replacement="    def compute(self):\n        return 20",
        )
        assert res_crlf.get("status") == "success"
        assert "return 20" in file_path.read_text(encoding="utf-8")

        # 2. Leading/trailing indentation difference fallback
        res_indent = edit_file(
            str(file_path),
            target="def compute(self):   \n  return 20",
            replacement="    def compute(self):\n        return 30",
        )
        assert res_indent.get("status") == "success"
        assert "return 30" in file_path.read_text(encoding="utf-8")

        # 3. Multiple fuzzy matches must be rejected for safety
        ambiguous_file = tmp_path / "ambiguous.py"
        ambiguous_file.write_text(
            "def first():\n    val = 42\n    return val\n\ndef second():\n    val = 42\n    return val\n",
            encoding="utf-8",
        )
        res_ambiguous = edit_file(
            str(ambiguous_file),
            target="val = 42\nreturn val",
            replacement="val = 99\nreturn val",
        )
        assert "occurs 2 times" in res_ambiguous.get("error", "")
        assert "val = 99" not in ambiguous_file.read_text(encoding="utf-8")

    def test_list_directory(self, tmp_path: Path):
        (tmp_path / "a.txt").write_text("a", encoding="utf-8")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.txt").write_text("b", encoding="utf-8")

        res = list_directory(str(tmp_path), max_depth=2)
        tree = res.get("tree", "")
        assert "📄 a.txt" in tree
        assert "📁 sub/" in tree
        assert "📄 b.txt" in tree

    def test_search_code(self, tmp_path: Path):
        (tmp_path / "main.py").write_text("class VertexClient:\n    pass\n", encoding="utf-8")
        (tmp_path / "other.py").write_text("def dummy():\n    return 42\n", encoding="utf-8")

        res = search_code("VertexClient", path=str(tmp_path))
        assert res.get("total_matches") == 1
        assert any("VertexClient" in m for m in res.get("matches", []))

    def test_run_command_success_and_timeout(self, tmp_path: Path):
        # Successful command
        res = run_command("echo 'test-output'", cwd=str(tmp_path))
        assert res.get("exit_code") == 0
        assert "test-output" in res.get("stdout", "")

        # Timeout command
        timeout_res = run_command("sleep 5", timeout_seconds=1)
        assert "timed out" in timeout_res.get("error", "")

    def test_execute_tool_dispatcher(self, tmp_path: Path):
        file_path = tmp_path / "dispatch.txt"
        res = execute_tool("write_file", {"path": str(file_path), "content": "dispatched!"})
        assert res.get("status") == "success"

        unregistered = execute_tool("invalid_tool_name", {})
        assert "is not registered" in unregistered.get("error", "")

    def test_tools_edge_cases_and_validations(self, tmp_path: Path):
        # 1. read_file on a directory
        dir_res = read_file(str(tmp_path))
        assert "not a file" in dir_res.get("error", "")

        # 2. list_directory on nonexistent path and file
        fake_path = tmp_path / "ghost_folder"
        assert "does not exist" in list_directory(str(fake_path)).get("error", "")

        some_file = tmp_path / "f.txt"
        some_file.write_text("hello", encoding="utf-8")
        assert "not a directory" in list_directory(str(some_file)).get("error", "")

        # 3. search_code on nonexistent path
        assert "does not exist" in search_code("pattern", path=str(fake_path)).get("error", "")

        # 4. save_memory and search_memory through execute_tool
        save_res = execute_tool(
            "save_memory", {"title": "Dispatch Test", "content": "Memory content"}
        )
        assert save_res.get("status") == "saved"

        search_res = execute_tool("search_memory", {"query": "Dispatch"})
        assert search_res.get("count", 0) >= 1

        # 5. AST tools validations
        outline_dir = execute_tool("get_symbol_outline", {"file_path": str(tmp_path)})
        assert "not a file" in outline_dir.get("error", "")

        def_res = execute_tool("find_symbol_definition", {"symbol_name": "NonExistentClass"})
        assert def_res.get("count") == 0

        ref_res = execute_tool("find_symbol_references", {"symbol_name": "NonExistentClass"})
        assert ref_res.get("count") == 0

        impact_res = execute_tool("analyze_symbol_impact", {"symbol_name": "NonExistentClass"})
        assert impact_res.get("risk_rating") == "LOW"
        assert impact_res.get("reference_count") == 0

    def test_run_command_head_tail_windowing_and_secret_redaction(self, tmp_path: Path):
        # 1. Head-tail windowing when output > 10,000 chars
        script = tmp_path / "gen_output.py"
        script.write_text(
            'import sys\nsys.stdout.write("H" * 2000 + "M" * 5000 + "T" * 8000)\n',
            encoding="utf-8",
        )
        res = run_command(f"python3 {script}", cwd=str(tmp_path))
        assert res.get("exit_code") == 0
        assert res.get("truncated") is True
        stdout = res.get("stdout", "")
        assert stdout.startswith("H" * 2000)
        assert stdout.endswith("T" * 8000)
        assert "\n[... Output truncated: 5000 bytes omitted ...]\n" in stdout

        # 2. Secret redaction (Bearer tokens, gcloud tokens, private keys, API keys)
        secret_script = tmp_path / "gen_secrets.py"
        secret_script.write_text(
            "import sys\n"
            'print("Authorization: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.secretpayload")\n'
            'print("gcloud_token=ya29.a0AfH6SMBx999999999999999")\n'
            'print("-----BEGIN PRIVATE KEY-----\\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC\\n-----END PRIVATE KEY-----")\n'
            'print(\'{"private_key": "-----BEGIN RSA PRIVATE KEY-----\\\\nabc\\\\n-----END RSA PRIVATE KEY-----"}\')\n'
            'print("api_key=sk-1234567890abcdef12345678")\n',
            encoding="utf-8",
        )
        sec_res = run_command(f"python3 {secret_script}", cwd=str(tmp_path))
        sec_out = sec_res.get("stdout", "")
        assert "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9" not in sec_out
        assert "[REDACTED_TOKEN]" in sec_out
        assert "ya29.a0AfH6SMBx999999999999999" not in sec_out
        assert "[REDACTED_GCLOUD_TOKEN]" in sec_out
        assert "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC" not in sec_out
        assert "[REDACTED_PRIVATE_KEY]" in sec_out
        assert "sk-1234567890abcdef12345678" not in sec_out
        assert "[REDACTED_SECRET]" in sec_out
