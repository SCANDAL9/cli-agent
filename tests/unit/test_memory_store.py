"""
Unit tests for MemoryStore (FTS5 search, CRUD, ranking, edge cases).
"""

import pytest

from cli_agent.memory import MemoryStore, resolve_project_key


@pytest.mark.unit
class TestMemoryStore:
    def test_save_and_retrieve_recent_memories(self, temp_memory_store: MemoryStore):
        store = temp_memory_store

        id1 = store.save_memory(
            title="Architecture Note", content="Use hexagonal design", memory_type="decision"
        )
        id2 = store.save_memory(title="Bug Fix", content="Fixed N+1 query", memory_type="bugfix")

        assert id1 is not None and id2 is not None
        recent = store.get_recent_memories(limit=5)
        assert len(recent) == 2
        # Order is DESC by id
        assert recent[0]["id"] == id2
        assert recent[0]["title"] == "Bug Fix"
        assert recent[1]["id"] == id1

    def test_fts5_search_matches_content_and_title(self, temp_memory_store: MemoryStore):
        store = temp_memory_store

        store.save_memory(
            title="FastAPI integration",
            content="Configured ASGI server with Uvicorn workers",
            memory_type="config",
        )
        store.save_memory(
            title="Database choice",
            content="Decided to use SQLite with FTS5 for lightweight persistence",
            memory_type="decision",
        )

        results_fastapi = store.search_memory("FastAPI")
        assert len(results_fastapi) == 1
        assert results_fastapi[0]["title"] == "FastAPI integration"

        results_fts = store.search_memory("lightweight persistence")
        assert len(results_fts) == 1
        assert results_fts[0]["title"] == "Database choice"

    def test_special_characters_in_search_query_do_not_crash(self, temp_memory_store: MemoryStore):
        store = temp_memory_store
        store.save_memory(title="Special & Symbols", content="Regex testing (.*) and quotes ' \"")

        # Query with symbols that would break raw FTS5 syntax
        results = store.search_memory("Special & (.*) ' \"")
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_search_memory_fallback_to_like_when_no_alnum(self, temp_memory_store: MemoryStore):
        store = temp_memory_store
        store.save_memory(title="C++ Pointers", content="Dereferencing with *ptr")

        # Query with only non-alnum characters
        results = store.search_memory("*ptr")
        assert isinstance(results, list)

    def test_search_memory_empty_result(self, temp_memory_store: MemoryStore):
        store = temp_memory_store
        store.save_memory(title="Python Agent", content="CLI assistant")

        results = store.search_memory("NonExistentTermXYZ")
        assert results == []

    def test_project_scoped_isolation(self, temp_memory_store: MemoryStore):
        store = temp_memory_store

        # 1. Memory for Project Alpha
        store.save_memory(
            title="Backend Auth",
            content="Alpha uses JWT with RSA keys",
            project_key="project-alpha",
            scope="project",
        )

        # 2. Memory for Project Beta
        store.save_memory(
            title="Backend Auth",
            content="Beta uses Session Cookies with Redis",
            project_key="project-beta",
            scope="project",
        )

        # 3. Global developer preference
        store.save_memory(
            title="Code Style",
            content="Always use Google docstrings and Ruff formatter",
            project_key="global",
            scope="global",
        )

        # Query scoped to Alpha
        alpha_results = store.search_memory("Auth", project_key="project-alpha")
        assert len(alpha_results) == 1
        assert "Alpha uses JWT" in alpha_results[0]["content"]

        # Query scoped to Beta
        beta_results = store.search_memory("Auth", project_key="project-beta")
        assert len(beta_results) == 1
        assert "Beta uses Session Cookies" in beta_results[0]["content"]

        # Global memories should appear in both workspaces
        alpha_global = store.search_memory("Code Style", project_key="project-alpha")
        assert len(alpha_global) == 1
        assert "Always use Google docstrings" in alpha_global[0]["content"]

    def test_resolve_project_key_helper(self, tmp_path):
        # Fallback to dir name when not a git repo
        key = resolve_project_key(cwd=tmp_path)
        assert key == tmp_path.name
