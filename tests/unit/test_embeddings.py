from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from cli_agent.embeddings import FastEmbedEmbeddingProvider


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fastembed_provider_embeds_queries_and_batches_locally() -> None:
    model = Mock()
    model.query_embed.return_value = iter([[0.1, 0.2, 0.3]])
    model.embed.return_value = iter([[0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    provider = FastEmbedEmbeddingProvider(model_name="BAAI/bge-small-en-v1.5")
    with patch("fastembed.TextEmbedding", return_value=model) as model_class:
        assert await provider.embed_query("find authentication") == [0.1, 0.2, 0.3]
        assert await provider.embed_texts(["first", "second"]) == [
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ]
    model.query_embed.assert_called_once_with("find authentication")
    model.embed.assert_called_once_with(["first", "second"])
    model_class.assert_called_once_with(
        model_name="BAAI/bge-small-en-v1.5",
        cache_dir=str(Path.home() / ".cache" / "fastembed"),
        local_files_only=True,
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fastembed_provider_empty_inputs_do_not_load_model() -> None:
    provider = FastEmbedEmbeddingProvider()
    assert await provider.embed_texts([]) == []
    with patch("fastembed.TextEmbedding") as model:
        assert await provider.embed_query("") == []
    model.assert_not_called()
