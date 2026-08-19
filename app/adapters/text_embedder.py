import hashlib
from collections.abc import Sequence
from typing import Protocol

from langchain.embeddings import Embeddings
from langchain_classic.embeddings.cache import CacheBackedEmbeddings
from langchain_core.stores import InMemoryStore
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

from app.domain.models import Embedding
from app.services.ports import TextEmbedder

CACHE_NAMESPACE = "rag_cache"


class LangchainEmbedder(TextEmbedder, Protocol):
    @property
    def embedder(self) -> Embeddings: ...


class LangchainEmbedderBase:
    _embedder: Embeddings
    model_id: str

    def embed_texts(self, texts: Sequence[str]) -> list[Embedding]:
        return [
            Embedding(vector=tuple(vector), model_id=self.model_id)
            for vector in self._embedder.embed_documents(texts=list(texts))
        ]

    def embed_query(self, text: str) -> Embedding:
        vector = self._embedder.embed_query(text=text)
        return Embedding(vector=tuple(vector), model_id=self.model_id)


class LangChainOpenAITextEmbedder(LangchainEmbedderBase):
    def __init__(
        self, model_id: str, base_url: str, api_key: str, dimensions: int
    ) -> None:
        self._embedder = OpenAIEmbeddings(
            model=model_id,
            base_url=base_url,
            api_key=SecretStr(api_key),
            dimensions=dimensions,
            check_embedding_ctx_length=False,
        )
        self.model_id = model_id

    @property
    def embedder(self) -> Embeddings:
        return self._embedder


class LangChainOllamaTextEmbedder(LangchainEmbedderBase):
    def __init__(self, model_id: str, base_url: str, dimensions: int) -> None:
        self._embedder = OllamaEmbeddings(
            model=model_id, base_url=base_url, dimensions=dimensions
        )
        self.model_id = model_id

    @property
    def embedder(self) -> Embeddings:
        return self._embedder


# TODO: Will be replaced by Redis in the future.
class LangChainInMemoryCacheBackedEmbedder(LangchainEmbedderBase):
    def __init__(self, langchain_embedder: LangchainEmbedder) -> None:
        cache_store = InMemoryStore()

        self._embedder = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings=langchain_embedder.embedder,
            document_embedding_cache=cache_store,
            query_embedding_cache=True,
            key_encoder=self._sha256_encoder,
        )
        self.model_id = langchain_embedder.model_id

    @staticmethod
    def _sha256_encoder(key: str) -> str:
        namespaced_key = f"{CACHE_NAMESPACE}:{key}"
        return hashlib.sha256(namespaced_key.encode("utf-8")).hexdigest()

    @property
    def embedder(self) -> Embeddings:
        return self._embedder
