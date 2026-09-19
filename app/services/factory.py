import logging

import httpx

from app.adapters.file_reader import FileReaderRegistry, MarkdownReader, PdfReader
from app.adapters.llm import LangChainOpenAILLMChat, LangchainPromptTemplate
from app.adapters.security import (
    LocalPresidioPIIRedactor,
    LocalPresidioRegexOutputValidator,
    RegexInputSanitizer,
)
from app.adapters.text_embedder import (
    LangchainEmbedder,
    LangChainInMemoryCacheBackedEmbedder,
    LangChainOllamaTextEmbedder,
    LangChainOpenAITextEmbedder,
)
from app.adapters.text_splitter import LangChainRecursiveTextSplitter
from app.configs import get_settings
from app.exceptions import ValidationError
from app.observability import create_langchain_callback_handler, get_langfuse_client
from app.services.ports import (
    InputSanitizer,
    LLMChat,
    OutputValidator,
    PIIRedactor,
    TextEmbedder,
    TextSplitter,
)
from app.services.rag_service import (
    DocumentIngestionService,
    FileReaderService,
    InputValidatorService,
    RagService,
    WorkspaceService,
)
from app.services.uow import SQLAlchemyUnitOfWork, UnitOfWork

logger = logging.getLogger(__name__)


def _create_langchain_embedder(
    provider: str, model_id: str, base_url: str, dimensions: int
) -> LangchainEmbedder:
    langfuse_client = get_langfuse_client()
    if provider == "ollama":
        embedder: LangchainEmbedder = LangChainOllamaTextEmbedder(
            model_id=model_id,
            base_url=base_url,
            dimensions=dimensions,
            langfuse_client=langfuse_client,
        )
    elif provider == "openai":
        embedder: LangchainEmbedder = LangChainOpenAITextEmbedder(
            model_id=model_id,
            base_url=base_url,
            api_key=get_settings().OPENAI_API_KEY,
            dimensions=dimensions,
            langfuse_client=langfuse_client,
        )
    else:
        raise ValidationError(f"Unknown embedding provider: {provider}")

    return LangChainInMemoryCacheBackedEmbedder(
        embedder, langfuse_client=langfuse_client
    )


def _create_llm(
    provider: str, model_id: str, base_url: str, system_prompt: str
) -> LLMChat:
    if provider == "openai":
        callback_handler = create_langchain_callback_handler()
        callbacks = [callback_handler] if callback_handler else None
        return LangChainOpenAILLMChat(
            model_id=model_id,
            base_url=base_url,
            api_key=get_settings().OPENAI_API_KEY,
            max_retries=10,
            system_prompt=system_prompt,
            callbacks=callbacks,
        )
    raise ValidationError(f"Unknown LLM provider: {provider}")


def _ping(url: str) -> bool:
    try:
        httpx.get(url, timeout=5)
        return True
    except Exception:  # noqa
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

    primary = _create_llm(
        s.LLM_PROVIDER, s.LLM_MODEL, s.LLM_BASE_URL, s.LLM_SYSTEM_PROMPT
    )
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
        fb = _create_llm(fb_provider, fb_model, fb_url, s.LLM_SYSTEM_PROMPT)
        if _ping(fb_url):
            return fb

    logger.warning("All LLM providers unhealthy, returning primary anyway")
    return primary


def create_text_splitter() -> TextSplitter:
    s = get_settings()
    return LangChainRecursiveTextSplitter(
        chunk_size=s.CHUNK_SIZE, chunk_overlap=s.CHUNK_OVERLAP
    )


def create_file_reader_registry() -> FileReaderRegistry:
    registry = FileReaderRegistry()
    registry.register("pdf", PdfReader())
    registry.register("md", MarkdownReader())
    registry.register("markdown", MarkdownReader())
    return registry


def create_file_reader() -> FileReaderService:
    registry = create_file_reader_registry()
    return FileReaderService(registry)


def create_input_sanitizer() -> InputSanitizer:
    return RegexInputSanitizer()


def create_pii_redactor() -> PIIRedactor:
    return LocalPresidioPIIRedactor()


def create_input_validator() -> InputValidatorService:
    input_sanitizer = create_input_sanitizer()
    pii_redactor = create_pii_redactor()
    return InputValidatorService(input_sanitizer, pii_redactor)


def create_output_validator() -> OutputValidator:
    return LocalPresidioRegexOutputValidator()


def create_uow() -> UnitOfWork:
    return SQLAlchemyUnitOfWork()


def create_workspace_service(uow: UnitOfWork | None = None) -> WorkspaceService:
    return WorkspaceService(uow=uow or create_uow())


def create_ingestion_service(uow: UnitOfWork | None = None) -> DocumentIngestionService:
    return DocumentIngestionService(
        uow=uow or create_uow(),
        splitter=create_text_splitter(),
        embedder=create_text_embedder(),
    )


def create_rag_service(
    uow: UnitOfWork | None = None,
    embedder: TextEmbedder | None = None,
    llm: LLMChat | None = None,
) -> RagService:
    s = get_settings()
    return RagService(
        uow=uow or create_uow(),
        embedder=embedder or create_text_embedder(),
        llm=llm or create_llm_chat(),
        top_k=s.TOP_K,
        chat_prompt=LangchainPromptTemplate(s.RAG_PROMPT_TEMPLATE),
    )
