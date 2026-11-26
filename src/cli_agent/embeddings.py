"""Local FastEmbed adapters used by hybrid code search."""

import asyncio
from collections.abc import Iterable
from pathlib import Path
from typing import Any


class FastEmbedEmbeddingProvider:
    """Generate query/document vectors from a locally cached FastEmbed model.

    Model access is lazy and local-only: the normal search path never downloads
    model weights or makes a network request. If the model is not cached yet,
    callers can still use the sparse FTS5 path without a dense vector.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        cache_dir: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir or str(Path.home() / ".cache" / "fastembed")
        self._model: Any | None = None

    def _get_model(self) -> Any:
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(
                model_name=self.model_name,
                cache_dir=self.cache_dir,
                local_files_only=True,
            )
        return self._model

    @staticmethod
    def _to_vectors(rows: Iterable[Any]) -> list[list[float]]:
        return [
            row.tolist() if hasattr(row, "tolist") else [float(value) for value in row]
            for row in rows
        ]

    async def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            return []
        try:
            model = await asyncio.to_thread(self._get_model)
            vectors = await asyncio.to_thread(lambda: self._to_vectors(model.query_embed(text)))
            return vectors[0] if vectors else []
        except Exception:
            # A missing local model must not trigger a download or disable keyword search.
            return []

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            model = await asyncio.to_thread(self._get_model)
            return await asyncio.to_thread(lambda: self._to_vectors(model.embed(texts)))
        except Exception:
            # Store chunks for FTS5 even when users have not provisioned model weights.
            return [[] for _ in texts]
