import hashlib
import logging
from collections.abc import Sequence
from typing import Protocol

from langchain.embeddings import Embeddings as LangChainEmbeddings
from langchain_classic.embeddings.cache import CacheBackedEmbeddings
from langchain_core.stores import InMemoryStore
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from openai import RateLimitError as OpenAIRateLimitError
from pydantic import SecretStr

from app.configs import get_settings
from app.domain.models import Embedding
from app.exceptions import EmbeddingError, RateLimitError
from app.services.ports import TextEmbedder

logger = logging.getLogger(__name__)


class LangchainEmbedder(TextEmbedder, Protocol):
    @property
    def embedder(self) -> LangChainEmbeddings: ...


class LangChainEmbedderBase:
    _embedder: LangChainEmbeddings
    model_id: str

    def embed_texts(self, texts: Sequence[str]) -> list[Embedding]:
        try:
            return [
                Embedding(vector=tuple(vector), model_id=self.model_id)
                for vector in self._embedder.embed_documents(texts=list(texts))
            ]
        except OpenAIRateLimitError as e:
            logger.warning("Embedding rate limit exceeded: %s", e)
            raise RateLimitError() from e
        except Exception as e:  # noqa: BLE001
            logger.error("Embedding failed: %s", e)
            raise EmbeddingError("External api error")

    def embed_query(self, text: str) -> Embedding:
        try:
            vector = self._embedder.embed_query(text=text)
            return Embedding(vector=tuple(vector), model_id=self.model_id)
        except OpenAIRateLimitError as e:
            logger.warning("Embedding rate limit exceeded: %s", e)
            raise RateLimitError() from e
        except Exception as e:  # noqa: BLE001
            logger.error("Embedding failed: %s", e)
            raise EmbeddingError("External api error")


class LangChainOpenAITextEmbedder(LangChainEmbedderBase):
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
    def embedder(self) -> LangChainEmbeddings:
        return self._embedder


class LangChainOllamaTextEmbedder(LangChainEmbedderBase):
    def __init__(self, model_id: str, base_url: str, dimensions: int) -> None:
        self._embedder = OllamaEmbeddings(
            model=model_id, base_url=base_url, dimensions=dimensions
        )
        self.model_id = model_id

    @property
    def embedder(self) -> LangChainEmbeddings:
        return self._embedder


# TODO: Will be replaced by Redis in the future.
class LangChainInMemoryCacheBackedEmbedder(LangChainEmbedderBase):
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
        namespaced_key = f"{get_settings().CACHE_NAMESPACE}:{key}"
        return hashlib.sha256(namespaced_key.encode("utf-8")).hexdigest()

    @property
    def embedder(self) -> LangChainEmbeddings:
        return self._embedder
