from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import Any

from langchain.embeddings import Embeddings as LangChainEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langfuse import Langfuse, observe
from openai import APITimeoutError
from pydantic import SecretStr

from app.adapters.llm import OpenAIRateLimitError
from app.configs import get_settings
from app.domain.models import Embedding
from app.exceptions import EmbeddingError, EmbeddingTimeoutError, RateLimitError


class LangChainEmbedderBase:
    _embedder: LangChainEmbeddings
    _langfuse_client: Langfuse | None = None
    model_id: str

    def __init__(
        self,
        langfuse_client: Langfuse | None = None,
        max_batch_size: int | None = None,
    ) -> None:
        self._langfuse_client = langfuse_client
        self.max_batch_size = (
            max_batch_size
            if max_batch_size is not None
            else get_settings().EMBEDDING_MAX_BATCH_SIZE
        )

        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")

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

    @observe(
        name="text-embedding",
        as_type="embedding",
        capture_input=False,
        capture_output=False,
    )
    def embed_texts(self, texts: Sequence[str]) -> list[Embedding]:
        batches = self._batch_texts(texts)
        embeddings = []
        for batch in batches:
            try:
                vectors = self._embedder.embed_documents(texts=batch)
            except OpenAIRateLimitError as e:
                raise RateLimitError() from e

            except APITimeoutError as e:
                raise EmbeddingTimeoutError() from e

            except Exception as e:
                raise EmbeddingError("External embedding API error") from e

            embeddings.extend(
                [
                    Embedding(vector=tuple(vector), model_id=self.model_id)
                    for vector in vectors
                ]
            )
        return embeddings

    @observe(
        name="query-embedding",
        as_type="embedding",
        capture_input=False,
        capture_output=False,
    )
    def embed_query(self, text: str) -> Embedding:
        try:
            vector = self._embedder.embed_query(text=text)
            return Embedding(vector=tuple(vector), model_id=self.model_id)
        except OpenAIRateLimitError as e:
            raise RateLimitError() from e

        except APITimeoutError as e:
            raise EmbeddingTimeoutError() from e

        except Exception as e:
            raise EmbeddingError() from e

    def _batch_texts(self, texts: Sequence[str]) -> list[list[str]]:
        num_chunks = len(texts)
        if num_chunks == 0:
            return []
        batched = []
        for i in range(0, num_chunks, self.max_batch_size):
            batched.append(list(texts[i : i + self.max_batch_size]))
        return batched


class LangChainOpenAITextEmbedder(LangChainEmbedderBase):
    def __init__(
        self,
        model_id: str,
        base_url: str,
        api_key: str,
        dimensions: int,
        max_retries: int | None = None,
        langfuse_client: Langfuse | None = None,
        max_batch_size: int | None = None,
    ) -> None:
        super().__init__(langfuse_client=langfuse_client, max_batch_size=max_batch_size)
        base_url_arg = base_url if base_url else None
        max_retries = (
            max_retries
            if max_retries is not None
            else get_settings().EMBEDDING_MAX_RETRIES
        )
        self._embedder = OpenAIEmbeddings(
            model=model_id,
            base_url=base_url_arg,
            api_key=SecretStr(api_key),
            dimensions=dimensions,
            check_embedding_ctx_length=False,
            chunk_size=self.max_batch_size,
            max_retries=max_retries,
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
        max_batch_size: int | None = None,
    ) -> None:
        super().__init__(langfuse_client=langfuse_client, max_batch_size=max_batch_size)
        base_url_arg = base_url if base_url else None
        self._embedder = OllamaEmbeddings(
            model=model_id,
            base_url=base_url_arg,
            dimensions=dimensions,
        )
        self.model_id = model_id

    @property
    def embedder(self) -> LangChainEmbeddings:
        return self._embedder
