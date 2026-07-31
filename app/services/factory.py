import hashlib
import logging

import httpx
from fastapi import Depends
from langchain_classic.embeddings.cache import CacheBackedEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.stores import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr
from sqlalchemy.orm import Session

from app.configs import get_settings
from app.database import get_db
from app.services.agent import RagService
from app.services.embedding import (
    DocumentEmbeddingService,
    DocumentIngestionPipeline,
    ExactMatchDeduplicator,
)
from app.services.repository import DocumentRepository

logger = logging.getLogger(__name__)

CACHE_NAMESPACE = "rag_cache"


def _sha256_encoder(key: str) -> str:
    namespaced_key = f"{CACHE_NAMESPACE}:{key}"
    return hashlib.sha256(namespaced_key.encode("utf-8")).hexdigest()


def _create_embeddings(provider: str, model: str, base_url: str) -> Embeddings:
    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=model, base_url=base_url, dimensions=get_settings().DIMENSIONS
        )
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=model,
            base_url=base_url,
            api_key=SecretStr(get_settings().OPENAI_API_KEY),
            dimensions=get_settings().DIMENSIONS,
            check_embedding_ctx_length=False,
        )
    raise ValueError(f"Unknown embedding provider: {provider}")


def _create_llm(provider: str, model: str, base_url: str) -> BaseChatModel:
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model, base_url=base_url)
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=SecretStr(get_settings().OPENAI_API_KEY),
            max_retries=get_settings().LLM_MAX_RETRIES,
        )
    raise ValueError(f"Unknown LLM provider: {provider}")


def _ping(url: str) -> bool:
    try:
        httpx.get(url, timeout=5)
        return True
    except Exception:
        return False


def _get_embeddings_provider() -> Embeddings:
    s = get_settings()

    primary = _create_embeddings(
        s.EMBEDDING_PROVIDER, s.EMBEDDING_MODEL, s.EMBEDDING_BASE_URL
    )
    if _ping(s.EMBEDDING_BASE_URL):
        return primary

    logger.warning("Primary embedding provider unhealthy, trying fallback")

    fb_provider = s.EMBEDDING_FALLBACK_PROVIDER or s.EMBEDDING_PROVIDER
    fb_model = s.EMBEDDING_FALLBACK_MODEL or s.EMBEDDING_MODEL
    fb_url = s.EMBEDDING_FALLBACK_BASE_URL or s.EMBEDDING_BASE_URL

    if (fb_provider, fb_model, fb_url) != (
        s.EMBEDDING_PROVIDER,
        s.EMBEDDING_MODEL,
        s.EMBEDDING_BASE_URL,
    ):
        fb = _create_embeddings(fb_provider, fb_model, fb_url)
        if _ping(fb_url):
            return fb

    logger.warning("All embedding providers unhealthy, returning primary anyway")
    return primary


def create_llm_provider() -> BaseChatModel:
    s = get_settings()

    primary = _create_llm(s.LLM_PROVIDER, s.LLM_MODEL, s.LLM_BASE_URL)
    if _ping(s.LLM_BASE_URL):
        return primary

    logger.warning("Primary LLM provider unhealthy, trying fallback")

    fb_provider = s.LLM_FALLBACK_PROVIDER or s.LLM_PROVIDER
    fb_model = s.LLM_FALLBACK_MODEL or s.LLM_MODEL
    fb_url = s.LLM_FALLBACK_BASE_URL or s.LLM_BASE_URL

    if (fb_provider, fb_model, fb_url) != (
        s.LLM_PROVIDER,
        s.LLM_MODEL,
        s.LLM_BASE_URL,
    ):
        fb = _create_llm(fb_provider, fb_model, fb_url)
        if _ping(fb_url):
            return fb

    logger.warning("All LLM providers unhealthy, returning primary anyway")
    return primary


def create_cache_embeddings_provider() -> Embeddings:
    underlying_embedder = _get_embeddings_provider()

    # TODO: Will be replaced by Redis in the future.
    cache_store = InMemoryStore()

    return CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings=underlying_embedder,
        document_embedding_cache=cache_store,
        query_embedding_cache=True,
        key_encoder=_sha256_encoder,
    )


def create_embedding_service(
    embeddings_model: Embeddings,
) -> DocumentEmbeddingService:
    return DocumentEmbeddingService(
        splitter=RecursiveCharacterTextSplitter(
            chunk_size=get_settings().CHUNK_SIZE,
            chunk_overlap=get_settings().CHUNK_OVERLAP,
        ),
        embeddings_model=embeddings_model,
    )


def get_document_repository(session: Session = Depends(get_db)) -> DocumentRepository:
    return DocumentRepository(session=session)


def create_ingestion_pipeline(
    embedding_service: DocumentEmbeddingService,
    repository: DocumentRepository,
) -> DocumentIngestionPipeline:
    return DocumentIngestionPipeline(
        embedding_service=embedding_service,
        repository=repository,
        deduplicator=ExactMatchDeduplicator(repository),
    )


def create_rag_service(
    embeddings_model: Embeddings,
    session: Session,
) -> RagService:
    return RagService(
        db_session=session,
        embeddings_model=embeddings_model,
        llm=create_llm_provider(),
        top_k=get_settings().TOP_K,
    )
