import hashlib
import logging
from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import Any, Protocol

from langchain.embeddings import Embeddings as LangChainEmbeddings
from langchain_classic.embeddings.cache import CacheBackedEmbeddings
from langchain_core.stores import InMemoryStore
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langfuse import Langfuse
from pydantic import SecretStr

from app.adapters.llm import OpenAIRateLimitError
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
    _langfuse_client: Langfuse | None = None
    model_id: str

    def __init__(self, langfuse_client: Langfuse | None = None) -> None:
        self._langfuse_client = langfuse_client

    def _trace_embedding(
        self,
        name: str,
        input_data: dict[str, Any],
    ) -> AbstractContextManager[Any]:
        if self._langfuse_client is None:
            return nullcontext(None)

        return self._langfuse_client.start_as_current_observation(
            as_type="embedding",
            name=name,
            input=input_data,
            model=self.model_id,
        )

    def embed_texts(self, texts: Sequence[str]) -> list[Embedding]:
        try:
            with self._trace_embedding(
                name="embed-documents",
                input_data={"documents": texts, "document_count": len(texts)},
            ) as observation:
                vectors = self._embedder.embed_documents(texts=list(texts))

                if observation is not None:
                    observation.update(
                        output={
                            "embedding_count": len(vectors),
                            "dimensions": len(vectors[0]) if vectors else 0,
                        }
                    )

                return [
                    Embedding(
                        vector=tuple(vector),
                        model_id=self.model_id,
                    )
                    for vector in vectors
                ]
        except OpenAIRateLimitError as e:
            logger.warning("Embedding rate limit exceeded: %s", e)
            raise RateLimitError() from e
        except Exception as e:
            logger.exception("Embedding failed")
            raise EmbeddingError("External api error") from e

    def embed_query(self, text: str) -> Embedding:
        try:
            with self._trace_embedding(
                name="embed-query",
                input_data={"query_text": text},
            ) as observation:
                vector = self._embedder.embed_query(text=text)

                if observation is not None:
                    observation.update(
                        output={
                            "dimensions": len(vector),
                        }
                    )

                return Embedding(
                    vector=tuple(vector),
                    model_id=self.model_id,
                )

        except OpenAIRateLimitError as e:
            logger.warning("Embedding rate limit exceeded: %s", e)
            raise RateLimitError() from e
        except Exception as e:
            logger.exception("Embedding failed")
            raise EmbeddingError("External api error") from e


class LangChainOpenAITextEmbedder(LangChainEmbedderBase):
    def __init__(
        self,
        model_id: str,
        base_url: str,
        api_key: str,
        dimensions: int,
        langfuse_client: Langfuse | None = None,
    ) -> None:
        super().__init__(langfuse_client=langfuse_client)

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
    def __init__(
        self,
        model_id: str,
        base_url: str,
        dimensions: int,
        langfuse_client: Langfuse | None = None,
    ) -> None:
        super().__init__(langfuse_client=langfuse_client)

        self._embedder = OllamaEmbeddings(
            model=model_id,
            base_url=base_url,
            dimensions=dimensions,
        )

        self.model_id = model_id

    @property
    def embedder(self) -> LangChainEmbeddings:
        return self._embedder


# TODO: Add a cache hit/miss in the tracing
# TODO: Will be replaced by Redis in the future.
class LangChainInMemoryCacheBackedEmbedder(LangChainEmbedderBase):
    def __init__(
        self,
        langchain_embedder: LangchainEmbedder,
        langfuse_client: Langfuse | None = None,
    ) -> None:
        super().__init__(langfuse_client=langfuse_client)

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
