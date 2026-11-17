from functools import cache

from app.adapters.file_reader import FileReaderRegistry, MarkdownReader, PdfReader
from app.adapters.llm import LangChainOpenAILLMChat, LangchainPromptTemplate
from app.adapters.retriever import BM25Retriever, HybridRetriever, PgVectorRetriever
from app.adapters.security import (
    LocalPresidioInputValidator,
    LocalPresidioOutputValidator,
)
from app.adapters.text_embedder import (
    LangchainEmbedder,
    LangChainInMemoryCacheBackedEmbedder,
    LangChainOllamaTextEmbedder,
    LangChainOpenAITextEmbedder,
)
from app.adapters.text_splitter import LangChainRecursiveTextSplitter
from app.configs import get_settings
from app.exceptions import UnsupportedProvider
from app.observability import create_langchain_callback_handler, get_langfuse_client
from app.services.ports import (
    InputValidator,
    LLMChat,
    OutputValidator,
    Retriever,
    TextEmbedder,
    TextSplitter,
)
from app.services.rag_service import (
    DocumentIngestionService,
    FileReaderService,
    RagService,
    WorkspaceService,
)
from app.services.uow import SQLAlchemyUnitOfWork, UnitOfWork


def _create_langchain_embedder(
    provider: str, model_id: str, base_url: str, dimensions: int
) -> LangchainEmbedder:
    langfuse_client = get_langfuse_client()
    match provider:
        case "ollama":
            embedder: LangchainEmbedder = LangChainOllamaTextEmbedder(
                model_id=model_id,
                base_url=base_url,
                dimensions=dimensions,
                langfuse_client=langfuse_client,
            )
        case "openai":
            embedder: LangchainEmbedder = LangChainOpenAITextEmbedder(
                model_id=model_id,
                base_url=base_url,
                api_key=get_settings().OPENAI_API_KEY,
                dimensions=dimensions,
                langfuse_client=langfuse_client,
            )
        case _:
            raise UnsupportedProvider()

    return LangChainInMemoryCacheBackedEmbedder(
        embedder, langfuse_client=langfuse_client
    )


def _create_llm(
    provider: str, model_id: str, base_url: str, system_prompt: str
) -> LLMChat:
    match provider:
        case "openai":
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
        case _:
            raise UnsupportedProvider()


@cache
def create_text_embedder() -> TextEmbedder:
    s = get_settings()
    return _create_langchain_embedder(
        s.EMBEDDING_PROVIDER,
        s.EMBEDDING_MODEL,
        s.EMBEDDING_BASE_URL,
        s.DIMENSIONS,
    )


@cache
def create_llm_chat() -> LLMChat:
    s = get_settings()
    return _create_llm(
        s.LLM_PROVIDER,
        s.LLM_MODEL,
        s.LLM_BASE_URL,
        s.LLM_SYSTEM_PROMPT,
    )


@cache
def create_text_splitter() -> TextSplitter:
    s = get_settings()
    return LangChainRecursiveTextSplitter(
        chunk_size=s.CHUNK_SIZE, chunk_overlap=s.CHUNK_OVERLAP
    )


@cache
def create_file_reader_registry() -> FileReaderRegistry:
    registry = FileReaderRegistry()
    registry.register("pdf", PdfReader())
    registry.register("md", MarkdownReader())
    registry.register("markdown", MarkdownReader())
    return registry


@cache
def create_file_reader() -> FileReaderService:
    registry = create_file_reader_registry()
    return FileReaderService(registry)


@cache
def create_input_validator() -> InputValidator:
    return LocalPresidioInputValidator()


@cache
def create_output_validator() -> OutputValidator:
    return LocalPresidioOutputValidator()


def create_uow() -> UnitOfWork:
    return SQLAlchemyUnitOfWork()


def create_workspace_service(uow: UnitOfWork | None = None) -> WorkspaceService:
    return WorkspaceService(uow=uow or create_uow())


def create_ingestion_service(uow: UnitOfWork | None = None) -> DocumentIngestionService:
    return DocumentIngestionService(
        uow=uow or create_uow(),
        splitter=create_text_splitter(),
        embedder=create_text_embedder(),
        input_validator=create_input_validator(),
    )


def create_dense_retriever() -> PgVectorRetriever:
    return PgVectorRetriever(uow=SQLAlchemyUnitOfWork())


def create_sparse_retriever() -> BM25Retriever:
    return BM25Retriever(SQLAlchemyUnitOfWork())


def create_hybrid_retriever() -> HybridRetriever:
    s = get_settings()
    return HybridRetriever(
        embedder=create_text_embedder(),
        dense_retriever=create_dense_retriever(),
        sparse_retriever=create_sparse_retriever(),
        top_k_single=s.TOP_K,
        smoothing_factor=s.SMOOTHING_FACTOR,
    )


def create_rag_service(
    uow: UnitOfWork | None = None,
    retriever: Retriever | None = None,
    llm: LLMChat | None = None,
) -> RagService:
    s = get_settings()
    return RagService(
        uow=uow or create_uow(),
        retriever=retriever or create_hybrid_retriever(),
        llm=llm or create_llm_chat(),
        top_k=s.TOP_K_HYBRID,
        chat_prompt=LangchainPromptTemplate(s.RAG_PROMPT_TEMPLATE),
        input_validator=create_input_validator(),
        output_validator=create_output_validator(),
    )
