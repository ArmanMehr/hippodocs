from typing import override

from langchain_core.embeddings import Embeddings

from app.adapters.text_embedder import LangChainEmbedderBase


class FakeLangChainEmbeddingProvider(Embeddings):
    def __init__(self):
        self.calls: list[list[str]] = []

    @override
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(int(text[:4]))] for text in texts]

    @override
    def embed_query(self, text: str) -> list[float]:
        return [float(int(text[:4]))]


class FakeEmbedder(LangChainEmbedderBase):
    def __init__(self, embedder: Embeddings, max_batch_size: int | None = None):
        super().__init__(max_batch_size=max_batch_size)
        self._embedder = embedder
        self.model_id = "test-model"


def test_large_documents_are_embedded_in_batches():
    max_batch_size = 100
    num_texts = 4000

    provider = FakeLangChainEmbeddingProvider()
    embedder = FakeEmbedder(embedder=provider, max_batch_size=max_batch_size)

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
