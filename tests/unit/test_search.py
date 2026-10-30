"""
Unit tests for the Hybrid Code Search Engine (W2.7).
Verifies AST-aware chunking, dense vector cosine similarity, Reciprocal Rank Fusion (RRF),
FTS5 BM25 retrieval, Vertex AI embedding fallback, and tool execution.
"""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from cli_agent.search import (
    CodeChunker,
    FastLocalEmbeddingProvider,
    HybridSearchEngine,
    VertexEmbeddingProvider,
    cosine_similarity,
    reciprocal_rank_fusion,
    sanitize_fts5_query,
    set_hybrid_search_engine,
)
from cli_agent.tools import execute_tool


@pytest.mark.unit
class TestSearchMathAndProviders:
    def test_cosine_similarity(self) -> None:
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

        # Edge cases: zero, empty, or mismatched lengths
        assert cosine_similarity([], []) == 0.0
        assert cosine_similarity([1.0], [1.0, 2.0]) == 0.0
        assert cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0

    def test_reciprocal_rank_fusion(self) -> None:
        bm25_ranked = [101, 102, 103]
        vector_ranked = [102, 104, 101]

        fused = reciprocal_rank_fusion(bm25_ranked, vector_ranked, k=60)
        # 102 is rank 1 in BM25 (idx 1, 1/62) and rank 0 in Vector (idx 0, 1/61)
        # 101 is rank 0 in BM25 (1/61) and rank 2 in Vector (1/63)
        # Expected: 102 has highest score
        top_id, top_score = fused[0]
        assert top_id == 102
        assert top_score == pytest.approx(1 / 62 + 1 / 61)

        # Empty rankings
        assert reciprocal_rank_fusion([], []) == []

    @pytest.mark.asyncio
    async def test_fast_local_embedding_provider(self) -> None:
        provider = FastLocalEmbeddingProvider(dimensions=32)

        q_vec = await provider.embed_query("token authentication")
        assert len(q_vec) == 32
        # Verify unit norm
        norm = sum(x * x for x in q_vec) ** 0.5
        assert norm == pytest.approx(1.0)

        # Verify determinism
        q_vec2 = await provider.embed_query("token authentication")
        assert q_vec == q_vec2

        # Empty string
        empty_vec = await provider.embed_query("")
        assert len(empty_vec) == 32
        assert all(x == 0.0 for x in empty_vec)

    @pytest.mark.asyncio
    async def test_vertex_embedding_provider_with_fallback(self) -> None:
        provider = VertexEmbeddingProvider()

        # Simulate exception in Vertex HTTP request -> triggers local fallback
        with patch("httpx.AsyncClient.post", side_effect=RuntimeError("GCloud network timeout")):
            emb = await provider.embed_query("search query")
            assert len(emb) == 64

        # Simulate valid Vertex API 200 response
        from unittest.mock import MagicMock

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "predictions": [{"embeddings": {"values": [0.1, 0.2, 0.3]}}]
        }

        with (
            patch.object(
                provider.token_provider, "get_token", AsyncMock(return_value="mock-token")
            ),
            patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response),
        ):
            embeddings = await provider.embed_texts(["hello world"])
            assert len(embeddings) == 1
            assert embeddings[0] == [0.1, 0.2, 0.3]


@pytest.mark.unit
class TestCodeChunker:
    def test_chunk_python_file(self, tmp_path: Path) -> None:
        code = """
import os

# Preamble comment
CONFIG_VALUE = 42

class AuthHandler:
    '''Class handling credentials.'''
    def __init__(self, token: str):
        self.token = token

    def validate(self) -> bool:
        return bool(self.token)

def helper_function(x: int) -> int:
    return x * 2
"""
        py_file = tmp_path / "auth.py"
        py_file.write_text(code, encoding="utf-8")

        chunks = CodeChunker.chunk_python_file(py_file, project_key="test_proj")
        assert len(chunks) >= 3

        symbol_names = {c.symbol_name for c in chunks}
        assert "AuthHandler" in symbol_names
        assert "AuthHandler.__init__" in symbol_names or "AuthHandler.validate" in symbol_names
        assert "helper_function" in symbol_names

        # Verify chunk types
        class_chunk = next(c for c in chunks if c.symbol_name == "AuthHandler")
        assert class_chunk.symbol_type == "class"

        func_chunk = next(c for c in chunks if c.symbol_name == "helper_function")
        assert func_chunk.symbol_type == "function"
        assert func_chunk.start_line == 15


@pytest.mark.unit
@pytest.mark.asyncio
class TestHybridSearchEngine:
    async def test_indexing_and_hybrid_search(self, tmp_path: Path) -> None:
        db_path = tmp_path / "search_test.db"
        engine = HybridSearchEngine(db_path=db_path)

        code_dir = tmp_path / "src"
        code_dir.mkdir()
        file1 = code_dir / "user_service.py"
        file1.write_text(
            """
class UserService:
    def authenticate_user(self, username: str, secret_key: str) -> bool:
        \"\"\"Validates user identity against secure credential store.\"\"\"
        return secret_key == "top_secret"

    def delete_account(self, user_id: str) -> None:
        pass
""",
            encoding="utf-8",
        )

        file2 = code_dir / "payment_gateway.py"
        file2.write_text(
            """
def process_credit_card(card_number: str, amount: float) -> str:
    \"\"\"Processes credit card transaction via billing provider.\"\"\"
    return "txn_12345"
""",
            encoding="utf-8",
        )

        indexed_count = await engine.index_directory(code_dir, project_key="test_app")
        assert indexed_count >= 3

        # 1. Search by exact keyword
        res_exact = await engine.search(query="authenticate_user", project_key="test_app", limit=2)
        assert len(res_exact) > 0
        assert "authenticate_user" in res_exact[0]["symbol_name"]
        assert res_exact[0]["rrf_score"] > 0

        # 2. Search by semantic intent
        res_semantic = await engine.search(
            query="credit card payment transaction billing", project_key="test_app", limit=2
        )
        assert len(res_semantic) > 0
        assert "process_credit_card" in res_semantic[0]["symbol_name"]

        # 3. Clean up and tool integration test
        set_hybrid_search_engine(engine)
        tool_res = await execute_tool("hybrid_search_code", {"query": "billing", "limit": 2})
        assert tool_res["count"] > 0
        assert "process_credit_card" in str(tool_res["results"])

        # Reset global search engine
        set_hybrid_search_engine(None)

    async def test_fts5_query_sanitization_and_resilience(self, tmp_path: Path) -> None:
        # Unit checks on sanitize_fts5_query with special FTS5 syntax chars
        q = sanitize_fts5_query('auth-token: "user" (admin)* AND OR NOT')
        assert '"auth"*' in q
        assert '"token"*' in q
        assert '"user"*' in q
        assert '"admin"*' in q
        assert sanitize_fts5_query('--- ::: "" () *') == ""

        db_path = tmp_path / "search_fts_sanitize.db"
        engine = HybridSearchEngine(db_path=db_path)

        code_dir = tmp_path / "src_sanitize"
        code_dir.mkdir()
        (code_dir / "handler.py").write_text(
            "def validate_jwt_token(token: str) -> bool:\n    return True\n",
            encoding="utf-8",
        )
        await engine.index_directory(code_dir, project_key="sanitize_proj")

        # Queries containing special characters that would crash raw FTS5
        for tricky_query in [
            'validate-jwt: "token" (str)*',
            "validate_jwt_token(token: str) -> bool",
            '--- : "" ( ) *',
        ]:
            results = await engine.search(query=tricky_query, project_key="sanitize_proj", limit=3)
            assert isinstance(results, list)
