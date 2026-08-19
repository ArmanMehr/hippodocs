import logging

import httpx

from app.adapters.llm import LangChainOpenAILLMChat
from app.adapters.text_embedder import (
    LangchainEmbedder,
    LangChainInMemoryCacheBackedEmbedder,
    LangChainOllamaTextEmbedder,
    LangChainOpenAITextEmbedder,
)
from app.adapters.text_splitter import LangChainRecursiveTextSplitter
from app.configs import get_settings
from app.services.ports import LLMChat, TextEmbedder, TextSplitter
from app.services.rag_service import (
    DocumentIngestionService,
    RagService,
    WorkspaceService,
)
from app.services.uow import SQLAlchemyUnitOfWork, UnitOfWork

logger = logging.getLogger(__name__)


def _create_langchain_embedder(
    provider: str, model_id: str, base_url: str, dimensions: int
) -> LangchainEmbedder:
    if provider == "ollama":
        embedder: LangchainEmbedder = LangChainOllamaTextEmbedder(
            model_id=model_id, base_url=base_url, dimensions=dimensions
        )
    elif provider == "openai":
        embedder: LangchainEmbedder = LangChainOpenAITextEmbedder(
            model_id=model_id,
            base_url=base_url,
            api_key=get_settings().OPENAI_API_KEY,
            dimensions=dimensions,
        )
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")

    return LangChainInMemoryCacheBackedEmbedder(embedder)


def _create_llm(provider: str, model_id: str, base_url: str) -> LLMChat:
    if provider == "openai":
        return LangChainOpenAILLMChat(
            model_id=model_id,
            base_url=base_url,
            api_key=get_settings().OPENAI_API_KEY,
            max_retries=10,
        )
    raise ValueError(f"Unknown LLM provider: {provider}")


def _ping(url: str) -> bool:
    try:
        httpx.get(url, timeout=5)
        return True
    except Exception:
        return False


def create_text_embedder() -> TextEmbedder:
    s = get_settings()

    primary = _create_langchain_embedder(
        s.EMBEDDING_PROVIDER, s.EMBEDDING_MODEL, s.EMBEDDING_BASE_URL, s.DIMENSIONS
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
        fb = _create_langchain_embedder(fb_provider, fb_model, fb_url, s.DIMENSIONS)
        if _ping(fb_url):
            return fb

    logger.warning("All embedding providers unhealthy, returning primary anyway")
    return primary


def create_llm_chat() -> LLMChat:
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


def create_text_splitter() -> TextSplitter:
    s = get_settings()
    return LangChainRecursiveTextSplitter(
        chunk_size=s.CHUNK_SIZE, chunk_overlap=s.CHUNK_OVERLAP
    )


def create_uow() -> UnitOfWork:
    return SQLAlchemyUnitOfWork()


def create_ingestion_service() -> DocumentIngestionService:
    return DocumentIngestionService(
        uow=create_uow(),
        splitter=create_text_splitter(),
        embedder=create_text_embedder(),
    )


def create_rag_service() -> RagService:
    return RagService(
        uow=create_uow(),
        embedder=create_text_embedder(),
        llm=create_llm_chat(),
        top_k=get_settings().TOP_K,
    )


def create_workspace_service() -> WorkspaceService:
    return WorkspaceService(uow=create_uow())
