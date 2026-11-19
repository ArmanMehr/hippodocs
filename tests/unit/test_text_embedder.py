from collections.abc import Sequence
from typing import override

from langchain_core.embeddings import Embeddings

from app.adapters.text_embedder import LangChainInMemoryCacheBackedEmbedder
from app.domain.models import Embedding


class FakeLangChainEmbeddingProvider(Embeddings):
    def __init__(self):
        self.calls: list[list[str]] = []
        self.query_calls = 0

    @override
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(int(text[:4]))] for text in texts]

    @override
    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return [float(int(text[:4]))]


class FakeLangchainEmbedder:
    model_id = "test-model"

    def __init__(self, embedder: Embeddings):
        self._embedder = embedder

    def embed_texts(self, texts: Sequence[str]) -> list[Embedding]:
        vectors = self._embedder.embed_documents(list(texts))
        return [Embedding(vector=tuple(vec)) for vec in vectors]

    def embed_query(self, text: str) -> Embedding:
        vector = self._embedder.embed_query(text)
        return Embedding(vector=tuple(vector))

    @property
    def embedder(self) -> Embeddings:
        return self._embedder


def test_large_documents_are_embedded_in_batches():
    max_batch_size = 100
    num_texts = 4000

    provider = FakeLangChainEmbeddingProvider()
    langchain_embedder = FakeLangchainEmbedder(embedder=provider)

    embedder = LangChainInMemoryCacheBackedEmbedder(
        langchain_embedder=langchain_embedder,
        max_batch_size=max_batch_size,
    )

    texts = [f"{i:04d}" + "x" * 100 for i in range(num_texts)]
    embeddings = embedder.embed_texts(texts)

    assert len(embeddings) == num_texts
    assert len(provider.calls) == num_texts // max_batch_size
    assert all(len(batch) <= max_batch_size for batch in provider.calls)

    processed_texts = [text for batch in provider.calls for text in batch]

    assert processed_texts == texts
    assert [int(embedding.vector[0]) for embedding in embeddings] == list(
        range(num_texts)
    )


def test_cache_backed_embedding_cache_hit_in_batch():
    num_texts = 100
    provider = FakeLangChainEmbeddingProvider()
    langchain_embedder = FakeLangchainEmbedder(embedder=provider)

    embedder = LangChainInMemoryCacheBackedEmbedder(
        langchain_embedder=langchain_embedder,
        max_batch_size=100,
    )

    texts = [f"{i:04d}" + "x" * 100 for i in range(num_texts)]
    embeddings = embedder.embed_texts(texts)
    embeddings2 = embedder.embed_texts(texts)

    # Cache hit
    assert len(provider.calls) == 1
    assert embeddings == embeddings2

    texts2 = [f"{i:04d}" + "x" * 100 for i in range(num_texts, num_texts * 2)]
    embeddings_miss = embedder.embed_texts(texts2)

    # Cache miss
    assert len(provider.calls) == 2
    assert embeddings_miss != embeddings


def test_cache_backed_embedding_cache_hit_in_query():
    provider = FakeLangChainEmbeddingProvider()
    langchain_embedder = FakeLangchainEmbedder(embedder=provider)

    embedder = LangChainInMemoryCacheBackedEmbedder(
        langchain_embedder=langchain_embedder
    )

    embedding = embedder.embed_query("1234query")
    embedding2 = embedder.embed_query("1234query")

    # Cache hit
    assert provider.query_calls == 1
    assert embedding == embedding2

    embedding_miss = embedder.embed_query("5678query")

    # Cache miss
    assert provider.query_calls == 2
    assert embedding_miss != embedding
