"""
Unit tests for the Tree-sitter AST & Code Intelligence Graph module.
"""

from pathlib import Path

import pytest

from cli_agent.code_graph import (
    CodeGraphIndexer,
    CodeGraphStore,
    CodeSymbol,
    SymbolEdge,
)
from cli_agent.tools import (
    analyze_symbol_impact,
    find_symbol_definition,
    find_symbol_references,
    get_symbol_outline,
    set_code_graph_store,
)


@pytest.fixture
def temp_graph_store(tmp_path: Path) -> CodeGraphStore:
    db_file = tmp_path / "test_graph.db"
    return CodeGraphStore(db_path=db_file)


@pytest.fixture
def temp_indexer(temp_graph_store: CodeGraphStore) -> CodeGraphIndexer:
    return CodeGraphIndexer(store=temp_graph_store)


@pytest.mark.unit
class TestCodeGraphStore:
    def test_save_and_retrieve_symbols(self, temp_graph_store: CodeGraphStore) -> None:
        sym = CodeSymbol(
            symbol_name="Calculator",
            symbol_type="class",
            start_line=1,
            end_line=20,
            signature="class Calculator(BaseEngine)",
            docstring="Performs math operations.",
        )
        edge = SymbolEdge(target_symbol_name="BaseEngine", relationship_type="inherits")

        temp_graph_store.save_file_symbols("proj1", "/src/calc.py", [(sym, [edge])])

        retrieved = temp_graph_store.get_file_symbols("proj1", "/src/calc.py")
        assert len(retrieved) == 1
        assert retrieved[0]["symbol_name"] == "Calculator"
        assert retrieved[0]["symbol_type"] == "class"
        assert "BaseEngine" in retrieved[0]["signature"]

        stats = temp_graph_store.get_stats("proj1")
        assert stats["files"] == 1
        assert stats["symbols"] == 1
        assert stats["edges"] == 1

    def test_find_definition_and_references(self, temp_graph_store: CodeGraphStore) -> None:
        sym_service = CodeSymbol(
            symbol_name="AuthService",
            symbol_type="class",
            start_line=10,
            end_line=50,
            signature="class AuthService",
            docstring="Handles login.",
        )
        sym_caller = CodeSymbol(
            symbol_name="login_route",
            symbol_type="function",
            start_line=60,
            end_line=75,
            signature="def login_route(req: Request) -> Response",
            docstring="Route handler.",
        )
        edge_call = SymbolEdge(target_symbol_name="AuthService", relationship_type="calls")

        temp_graph_store.save_file_symbols(
            "proj_auth",
            "/src/auth.py",
            [(sym_service, []), (sym_caller, [edge_call])],
        )

        defs = temp_graph_store.find_definition("proj_auth", "AuthService")
        assert len(defs) == 1
        assert defs[0]["symbol_name"] == "AuthService"
        assert defs[0]["start_line"] == 10

        refs = temp_graph_store.find_references("proj_auth", "AuthService")
        assert len(refs) == 1
        assert refs[0]["caller_name"] == "login_route"
        assert refs[0]["relationship_type"] == "calls"

    def test_get_symbol_impact_ratings(self, temp_graph_store: CodeGraphStore) -> None:
        sym_target = CodeSymbol(
            symbol_name="CoreService.execute",
            symbol_type="method",
            start_line=10,
            end_line=25,
            signature="def execute(self) -> None",
        )
        temp_graph_store.save_file_symbols("proj_imp", "/src/core.py", [(sym_target, [])])

        # 1. LOW risk (0 or 1 reference in 1 file)
        impact_low = temp_graph_store.get_symbol_impact("proj_imp", "CoreService.execute")
        assert impact_low["risk_rating"] == "LOW"
        assert impact_low["definition_count"] == 1
        assert impact_low["reference_count"] == 0

        # 2. MEDIUM risk (2 references across 2 files, testing qualified + short name matching)
        c1 = CodeSymbol("caller_a", "function", 1, 5)
        c2 = CodeSymbol("caller_b", "function", 1, 5)
        e1 = SymbolEdge("execute", "calls")
        e2 = SymbolEdge("CoreService.execute", "calls")
        temp_graph_store.save_file_symbols("proj_imp", "/src/mod_a.py", [(c1, [e1])])
        temp_graph_store.save_file_symbols("proj_imp", "/src/mod_b.py", [(c2, [e2])])

        impact_med = temp_graph_store.get_symbol_impact("proj_imp", "CoreService.execute")
        assert impact_med["risk_rating"] == "MEDIUM"
        assert impact_med["reference_count"] == 2
        assert impact_med["dependent_file_count"] == 2

        # 3. HIGH risk (>= 3 dependent files or >= 5 references)
        c3 = CodeSymbol("caller_c", "function", 1, 5)
        temp_graph_store.save_file_symbols("proj_imp", "/src/mod_c.py", [(c3, [e1])])

        impact_high = temp_graph_store.get_symbol_impact("proj_imp", "CoreService.execute")
        assert impact_high["risk_rating"] == "HIGH"
        assert impact_high["dependent_file_count"] == 3

    def test_get_outline_formatting(self, temp_graph_store: CodeGraphStore) -> None:
        sym_cls = CodeSymbol(
            symbol_name="UserService",
            symbol_type="class",
            start_line=5,
            end_line=30,
            signature="class UserService",
            docstring="User manager.",
        )
        sym_method = CodeSymbol(
            symbol_name="UserService.get_user",
            symbol_type="method",
            start_line=8,
            end_line=12,
            signature="def get_user(self, user_id: int) -> User",
            docstring="Fetches a user.",
        )
        temp_graph_store.save_file_symbols(
            "proj_ui", "/src/user.py", [(sym_cls, []), (sym_method, [])]
        )

        outline = temp_graph_store.get_outline("proj_ui", "/src/user.py")
        assert "### Structural Outline: /src/user.py" in outline
        assert "- [CLASS] class UserService (lines 5-30)" in outline
        assert '    """User manager."""' in outline
        assert "  - [METHOD] def get_user(self, user_id: int) -> User (lines 8-12)" in outline

        # Empty outline
        empty = temp_graph_store.get_outline("proj_ui", "/src/empty.py")
        assert "No indexed symbols found" in empty


@pytest.mark.unit
class TestCodeGraphIndexer:
    def test_parse_complex_python_source(self, temp_indexer: CodeGraphIndexer) -> None:
        code = '''
"""Module level docstring."""
import sys
from os.path import join

class DataPipeline(BasePipeline):
    """Pipeline processor."""

    def __init__(self, name: str) -> None:
        self.name = name

    def process_records(self, items: list[dict]) -> int:
        """Processes a list of records."""
        total = len(items)
        sys.stdout.write(f"Processing {total}")
        return total

def run_pipeline() -> None:
    """Entrypoint function."""
    pipe = DataPipeline("analytics")
    pipe.process_records([])
'''
        symbols = temp_indexer.parse_python_source(code, "pipeline.py", "proj_pipe")

        # Class
        classes = [s for s, _ in symbols if s.symbol_type == "class"]
        assert len(classes) == 1
        assert classes[0].symbol_name == "DataPipeline"
        assert classes[0].docstring == "Pipeline processor."

        # Methods
        methods = [s for s, _ in symbols if s.symbol_type == "method"]
        assert len(methods) == 2
        method_names = {m.symbol_name for m in methods}
        assert "DataPipeline.__init__" in method_names
        assert "DataPipeline.process_records" in method_names

        # Function
        functions = [s for s, _ in symbols if s.symbol_type == "function"]
        assert len(functions) == 1
        assert functions[0].symbol_name == "run_pipeline"
        assert functions[0].docstring == "Entrypoint function."

        # Calls inside run_pipeline
        run_edges = [edges for s, edges in symbols if s.symbol_name == "run_pipeline"][0]
        call_targets = {e.target_symbol_name for e in run_edges if e.relationship_type == "calls"}
        assert "DataPipeline" in call_targets
        assert "process_records" in call_targets

    def test_index_file_and_directory(
        self, temp_indexer: CodeGraphIndexer, temp_graph_store: CodeGraphStore, tmp_path: Path
    ) -> None:
        proj_dir = tmp_path / "project_root"
        proj_dir.mkdir()
        sub_dir = proj_dir / "pkg"
        sub_dir.mkdir()

        f1 = proj_dir / "mod1.py"
        f1.write_text("def helper(x: int) -> int:\n    return x * 2\n", encoding="utf-8")

        f2 = sub_dir / "mod2.py"
        f2.write_text(
            "class Worker:\n    def execute(self) -> None:\n        pass\n", encoding="utf-8"
        )

        # Ignored files
        ignore_dir = proj_dir / "__pycache__"
        ignore_dir.mkdir()
        (ignore_dir / "mod_cache.py").write_text("def cached(): pass\n", encoding="utf-8")

        txt_file = proj_dir / "readme.txt"
        txt_file.write_text("Not python code\n", encoding="utf-8")

        count = temp_indexer.index_directory(proj_dir, project_key="test_dir_proj")
        assert count == 2

        stats = temp_graph_store.get_stats("test_dir_proj")
        assert stats["files"] == 2
        assert stats["symbols"] >= 3

    def test_incremental_mtime_indexing_skips_unchanged_files(
        self,
        temp_indexer: CodeGraphIndexer,
        temp_graph_store: CodeGraphStore,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import os

        proj_dir = tmp_path / "inc_proj"
        proj_dir.mkdir()

        f1 = proj_dir / "alpha.py"
        f1.write_text("def alpha() -> int:\n    return 1\n", encoding="utf-8")

        f2 = proj_dir / "beta.py"
        f2.write_text("def beta() -> int:\n    return 2\n", encoding="utf-8")

        # First run indexes both files and stores st_mtime in file_metadata
        first_count = temp_indexer.index_directory(proj_dir, project_key="inc_key")
        assert first_count == 2
        assert temp_graph_store.get_file_mtime("inc_key", str(f1.resolve())) is not None
        assert temp_graph_store.get_file_mtime("inc_key", str(f2.resolve())) is not None

        # Second run with unchanged mtime must skip parsing completely
        parse_calls: list[str] = []
        orig_parse = temp_indexer.parse_python_source

        def spy_parse(code: str, file_path: str, project_key: str):
            parse_calls.append(file_path)
            return orig_parse(code, file_path, project_key)

        monkeypatch.setattr(temp_indexer, "parse_python_source", spy_parse)

        second_count = temp_indexer.index_directory(proj_dir, project_key="inc_key")
        assert second_count == 0
        assert len(parse_calls) == 0

        # Touch only f1 with a new mtime -> only f1 is re-indexed
        f1.write_text("def alpha() -> int:\n    return 100\n", encoding="utf-8")
        new_mtime = f1.stat().st_mtime + 10.0
        os.utime(f1, (new_mtime, new_mtime))

        third_count = temp_indexer.index_directory(proj_dir, project_key="inc_key")
        assert third_count == 1
        assert len(parse_calls) == 1
        assert str(f1.resolve()) in parse_calls[0]

        # Test explicit set_file_mtime helper
        temp_graph_store.set_file_mtime("inc_key", str(f2.resolve()), 12345.0)
        assert temp_graph_store.get_file_mtime("inc_key", str(f2.resolve())) == 12345.0


@pytest.mark.unit
class TestASTTools:
    def test_ast_tools_execution(
        self, temp_graph_store: CodeGraphStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        set_code_graph_store(temp_graph_store)
        monkeypatch.setattr("cli_agent.tools.resolve_project_key", lambda: "mock_ast_proj")

        py_file = tmp_path / "calc.py"
        py_file.write_text(
            'class MathEngine:\n    """Engine doc."""\n    def compute(self, a: int) -> int:\n        return a + 1\n',
            encoding="utf-8",
        )

        # Test get_symbol_outline
        res_outline = get_symbol_outline(str(py_file))
        assert "error" not in res_outline
        assert "MathEngine" in res_outline["outline"]
        assert "compute" in res_outline["outline"]

        # Test find_symbol_definition
        res_def = find_symbol_definition("MathEngine")
        assert res_def["count"] == 1
        assert res_def["definitions"][0]["symbol_type"] == "class"

        # Test analyze_symbol_impact
        res_impact = analyze_symbol_impact("MathEngine")
        assert res_impact["symbol_name"] == "MathEngine"
        assert res_impact["risk_rating"] == "LOW"
        assert res_impact["definition_count"] == 1

        # Test non-existent file error
        err_outline = get_symbol_outline(str(tmp_path / "does_not_exist.py"))
        assert "error" in err_outline

        # Test exception branches
        def raise_err(*args, **kwargs):
            raise RuntimeError("Database forced error")

        monkeypatch.setattr(temp_graph_store, "find_definition", raise_err)
        err_def = find_symbol_definition("MathEngine")
        assert "error" in err_def

        monkeypatch.setattr(temp_graph_store, "find_references", raise_err)
        err_ref = find_symbol_references("MathEngine")
        assert "error" in err_ref

        monkeypatch.setattr(temp_graph_store, "get_symbol_impact", raise_err)
        err_imp = analyze_symbol_impact("MathEngine")
        assert "error" in err_imp

        monkeypatch.setattr(temp_graph_store, "get_outline", raise_err)
        err_outline_ex = get_symbol_outline(str(py_file))
        assert "error" in err_outline_ex
