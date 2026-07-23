import logging
from functools import cache

import httpx
from fastapi import Depends
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
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


@cache
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


@cache
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


def get_embeddings_provider() -> Embeddings:
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

    if fb_provider != s.EMBEDDING_PROVIDER or fb_model != s.EMBEDDING_MODEL:
        fb = _create_embeddings(fb_provider, fb_model, fb_url)
        if _ping(fb_url):
            return fb

    logger.warning("All embedding providers unhealthy, returning primary anyway")
    return primary


def get_llm_provider() -> BaseChatModel:
    s = get_settings()

    primary = _create_llm(s.LLM_PROVIDER, s.LLM_MODEL, s.LLM_BASE_URL)
    if _ping(s.LLM_BASE_URL):
        return primary

    logger.warning("Primary LLM provider unhealthy, trying fallback")

    fb_provider = s.LLM_FALLBACK_PROVIDER or s.LLM_PROVIDER
    fb_model = s.LLM_FALLBACK_MODEL or s.LLM_MODEL
    fb_url = s.LLM_FALLBACK_BASE_URL or s.LLM_BASE_URL

    if fb_provider != s.LLM_PROVIDER or fb_model != s.LLM_MODEL:
        fb = _create_llm(fb_provider, fb_model, fb_url)
        if _ping(fb_url):
            return fb

    logger.warning("All LLM providers unhealthy, returning primary anyway")
    return primary


def get_embedding_service() -> DocumentEmbeddingService:
    return DocumentEmbeddingService(
        splitter=RecursiveCharacterTextSplitter(
            chunk_size=get_settings().CHUNK_SIZE,
            chunk_overlap=get_settings().CHUNK_OVERLAP,
        ),
        embeddings_model=get_embeddings_provider(),
    )


def get_document_repository(session: Session = Depends(get_db)) -> DocumentRepository:
    return DocumentRepository(session=session)


def get_ingestion_pipeline(
    repository: DocumentRepository = Depends(get_document_repository),
) -> DocumentIngestionPipeline:
    return DocumentIngestionPipeline(
        embedding_service=get_embedding_service(),
        repository=repository,
        deduplicator=ExactMatchDeduplicator(repository),
    )


def get_rag_service(session: Session = Depends(get_db)) -> RagService:
    return RagService(
        db_session=session,
        embeddings_model=get_embeddings_provider(),
        llm=get_llm_provider(),
        top_k=get_settings().TOP_K,
    )
