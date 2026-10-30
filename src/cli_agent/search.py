"""
Hybrid Code Search Engine for cli-agent.
Combines sparse keyword search (SQLite FTS5 BM25) with dense vector embeddings
via AST-aware code chunking and Reciprocal Rank Fusion (RRF).
"""

import hashlib
import json
import logging
import math
import re
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .code_graph import EXCLUDED_DIRS, PY_LANGUAGE
from .memory import DEFAULT_DB_PATH, resolve_project_key
from .token_provider import TokenProvider

logger = logging.getLogger(__name__)


@dataclass
class CodeChunk:
    """Represents an AST-coherent code chunk."""

    file_path: str
    symbol_name: str
    symbol_type: str
    start_line: int
    end_line: int
    content: str
    project_key: str
    embedding: list[float] | None = None
    id: int | None = None


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Calculates cosine similarity between two normalized or raw floating point vectors."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2, strict=False))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


def sanitize_fts5_query(query: str) -> str:
    """
    Sanitizes an arbitrary user search query into safe SQLite FTS5 quoted tokens.
    Strips special syntax characters (e.g., '-', ':', '\"', '(', ')', '*') and wraps
    clean alphanumeric/underscore tokens in double quotes to prevent FTS5 syntax crashes.
    """
    # Replace FTS5 operator/syntax characters with spaces and extract alphanumeric tokens
    cleaned = re.sub(r'[-:"()*^~+<>!?/\\|[\]{}@#$%&=,;.\'`]+', " ", query)
    tokens = [t for t in re.findall(r"[A-Za-z0-9_]+", cleaned) if t]
    if not tokens:
        return ""
    return " OR ".join(f'"{t}"*' for t in tokens)


def reciprocal_rank_fusion(
    bm25_ranked: list[int],
    vector_ranked: list[int],
    k: int = 60,
) -> list[tuple[int, float]]:
    """
    Computes Reciprocal Rank Fusion (RRF) scores across sparse and dense rankings.
    Formula: RRF_score(d) = sum(1 / (k + rank_m(d)))
    """
    scores: dict[int, float] = {}

    for rank, doc_id in enumerate(bm25_ranked):
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))

    for rank, doc_id in enumerate(vector_ranked):
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))

    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


class EmbeddingProvider(ABC):
    """Abstract interface for text embedding generation."""

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Embeds a single search query."""
        ...

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embeds a batch of code/text chunks."""
        ...


class FastLocalEmbeddingProvider(EmbeddingProvider):
    """
    Deterministic, zero-latency local embedding provider using hashed sub-word n-grams.
    Ensures immediate offline operation and zero external API dependencies for testing.
    """

    def __init__(self, dimensions: int = 64) -> None:
        self.dimensions = dimensions

    def _hash_embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dimensions
        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            return vec

        for token in tokens:
            # Hash token into dimension buckets
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dimensions
            weight = 1.0 + (len(token) / 10.0)
            vec[idx] += weight

        # Normalize to unit length
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    async def embed_query(self, text: str) -> list[float]:
        return self._hash_embed(text)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_embed(t) for t in texts]


class VertexEmbeddingProvider(EmbeddingProvider):
    """
    Vertex AI text-embedding-005 provider with graceful fallback to FastLocalEmbeddingProvider.
    """

    def __init__(
        self,
        project_id: str = "my-gcp-project",
        location: str = "us-central1",
        model: str = "text-embedding-005",
        token_provider: TokenProvider | None = None,
    ) -> None:
        self.project_id = project_id
        self.location = location
        self.model = model
        self.token_provider = token_provider or TokenProvider()
        self._fallback = FastLocalEmbeddingProvider()

    async def embed_query(self, text: str) -> list[float]:
        results = await self.embed_texts([text])
        return results[0] if results else [0.0] * 64

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        try:
            import httpx

            token = await self.token_provider.get_token()
            url = (
                f"https://{self.location}-aiplatform.googleapis.com/v1/"
                f"projects/{self.project_id}/locations/{self.location}/publishers/google/models/{self.model}:predict"
            )
            instances = [{"content": t, "task_type": "RETRIEVAL_DOCUMENT"} for t in texts]
            payload = {"instances": instances}

            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    url,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    preds = data.get("predictions", [])
                    return [p["embeddings"]["values"] for p in preds]
        except Exception as e:
            logger.debug(f"Vertex embedding call failed, falling back to local: {e}")

        return await self._fallback.embed_texts(texts)


class CodeChunker:
    """AST-aware code chunker leveraging tree-sitter."""

    @classmethod
    def chunk_python_file(cls, path: Path, project_key: str) -> list[CodeChunk]:
        """Parses a Python file into coherent AST chunks (classes, functions, methods)."""
        chunks: list[CodeChunk] = []
        try:
            code_bytes = path.read_bytes()
            code_str = code_bytes.decode("utf-8", errors="replace")
            lines = code_str.splitlines()
        except Exception:
            return chunks

        from tree_sitter import Parser

        parser = Parser(PY_LANGUAGE)
        tree = parser.parse(code_bytes)

        covered_lines: set[int] = set()

        def visit(node: Any, enclosing_class: str = "") -> None:
            if node.type in ("function_definition", "class_definition"):
                start_l = node.start_point[0] + 1
                end_l = node.end_point[0] + 1

                # Extract symbol name
                name_node = node.child_by_field_name("name")
                name = (
                    name_node.text.decode("utf-8", errors="replace")
                    if (name_node and name_node.text)
                    else "anonymous"
                )

                sym_type = "class" if node.type == "class_definition" else "function"
                if enclosing_class and sym_type == "function":
                    sym_type = "method"
                    name = f"{enclosing_class}.{name}"

                chunk_content = "\n".join(lines[start_l - 1 : end_l])
                chunks.append(
                    CodeChunk(
                        file_path=str(path),
                        symbol_name=name,
                        symbol_type=sym_type,
                        start_line=start_l,
                        end_line=end_l,
                        content=chunk_content,
                        project_key=project_key,
                    )
                )
                for line_idx in range(start_l, end_l + 1):
                    covered_lines.add(line_idx)

                # If class, visit methods
                if node.type == "class_definition":
                    body_node = node.child_by_field_name("body")
                    if body_node:
                        for child in body_node.children:
                            visit(child, enclosing_class=name)
                    return

            for child in node.children:
                visit(child, enclosing_class)

        visit(tree.root_node)

        # Catch top-level unchunked preamble if present
        uncovered = [idx for idx in range(1, len(lines) + 1) if idx not in covered_lines]
        if uncovered and len(lines) > 0:
            preamble_lines = lines[: min(uncovered[-1], 30)]
            if any(
                line_str.strip() and not line_str.strip().startswith("#")
                for line_str in preamble_lines
            ):
                chunks.append(
                    CodeChunk(
                        file_path=str(path),
                        symbol_name="<module_preamble>",
                        symbol_type="module",
                        start_line=1,
                        end_line=len(preamble_lines),
                        content="\n".join(preamble_lines),
                        project_key=project_key,
                    )
                )

        return chunks


class HybridSearchEngine:
    """
    Coordinates AST chunk indexing, SQLite FTS5 BM25 queries, dense vector scoring,
    and Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        db_path: Path | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.db_path = db_path or DEFAULT_DB_PATH
        self.embedding_provider = embedding_provider or FastLocalEmbeddingProvider()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS code_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_key TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    symbol_name TEXT NOT NULL,
                    symbol_type TEXT NOT NULL,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            cursor.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS code_chunks_fts USING fts5(
                    content,
                    symbol_name,
                    file_path,
                    content='code_chunks',
                    content_rowid='id'
                );
                """
            )
            conn.commit()

    async def index_file(self, file_path: Path, project_key: str) -> int:
        """Chunks and indexes a single file."""
        chunks = CodeChunker.chunk_python_file(file_path, project_key)
        if not chunks:
            return 0

        # Generate embeddings
        contents = [c.content for c in chunks]
        embeddings = await self.embedding_provider.embed_texts(contents)
        for chunk, emb in zip(chunks, embeddings, strict=False):
            chunk.embedding = emb

        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Remove prior chunks for this file
            cursor.execute(
                "DELETE FROM code_chunks WHERE project_key = ? AND file_path = ?",
                (project_key, str(file_path)),
            )
            cursor.execute(
                "DELETE FROM code_chunks_fts WHERE file_path = ?",
                (str(file_path),),
            )

            for chunk in chunks:
                cursor.execute(
                    """
                    INSERT INTO code_chunks (
                        project_key, file_path, symbol_name, symbol_type,
                        start_line, end_line, content, embedding
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.project_key,
                        chunk.file_path,
                        chunk.symbol_name,
                        chunk.symbol_type,
                        chunk.start_line,
                        chunk.end_line,
                        chunk.content,
                        json.dumps(chunk.embedding) if chunk.embedding else None,
                    ),
                )
                chunk_id = cursor.lastrowid
                cursor.execute(
                    """
                    INSERT INTO code_chunks_fts (rowid, content, symbol_name, file_path)
                    VALUES (?, ?, ?, ?)
                    """,
                    (chunk_id, chunk.content, chunk.symbol_name, chunk.file_path),
                )
            conn.commit()

        return len(chunks)

    async def index_directory(self, root_dir: Path | str, project_key: str) -> int:
        """Recursively indexes all Python files under root_dir."""
        base = Path(root_dir).resolve()
        total_chunks = 0
        for py_file in base.rglob("*.py"):
            if any(part in EXCLUDED_DIRS for part in py_file.parts):
                continue
            count = await self.index_file(py_file, project_key)
            total_chunks += count
        return total_chunks

    async def search(
        self,
        query: str,
        project_key: str | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Executes hybrid search (BM25 + Vector + Reciprocal Rank Fusion).
        """
        pk = project_key or resolve_project_key()

        # 1. Sparse Retrieval via SQLite FTS5 (BM25 ranking)
        bm25_ids: list[int] = []
        fts_query = sanitize_fts5_query(query)
        if fts_query:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute(
                        """
                        SELECT rowid FROM code_chunks_fts
                        WHERE code_chunks_fts MATCH ?
                        ORDER BY rank
                        LIMIT 50
                        """,
                        (fts_query,),
                    )
                    bm25_ids = [row[0] for row in cursor.fetchall()]
                except Exception as e:
                    logger.debug(f"FTS5 match failed: {e}")

        # 2. Dense Vector Retrieval
        query_vec = await self.embedding_provider.embed_query(query)
        vector_scores: list[tuple[int, float]] = []

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, embedding FROM code_chunks WHERE project_key = ?",
                (pk,),
            )
            for row in cursor.fetchall():
                c_id = row["id"]
                emb_raw = row["embedding"]
                if emb_raw:
                    try:
                        emb = json.loads(emb_raw)
                        sim = cosine_similarity(query_vec, emb)
                        vector_scores.append((c_id, sim))
                    except Exception:
                        pass

        vector_scores.sort(key=lambda x: x[1], reverse=True)
        vector_ids = [c_id for c_id, _ in vector_scores[:50]]

        # 3. Reciprocal Rank Fusion (RRF)
        fused = reciprocal_rank_fusion(bm25_ids, vector_ids, k=60)
        top_fused = fused[:limit]

        if not top_fused:
            return []

        # 4. Fetch Details for Top Results
        results: list[dict[str, Any]] = []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for doc_id, rrf_score in top_fused:
                cursor.execute(
                    """
                    SELECT id, file_path, symbol_name, symbol_type, start_line, end_line, content
                    FROM code_chunks WHERE id = ?
                    """,
                    (doc_id,),
                )
                row = cursor.fetchone()
                if row:
                    results.append(
                        {
                            "id": row["id"],
                            "file_path": row["file_path"],
                            "symbol_name": row["symbol_name"],
                            "symbol_type": row["symbol_type"],
                            "start_line": row["start_line"],
                            "end_line": row["end_line"],
                            "rrf_score": round(rrf_score, 5),
                            "content": row["content"],
                        }
                    )

        return results


# Global singleton instance for tools and REPL
_global_search_engine: HybridSearchEngine | None = None


def get_hybrid_search_engine() -> HybridSearchEngine:
    global _global_search_engine
    if _global_search_engine is None:
        _global_search_engine = HybridSearchEngine()
    return _global_search_engine


def set_hybrid_search_engine(engine: HybridSearchEngine | None) -> None:
    global _global_search_engine
    _global_search_engine = engine
