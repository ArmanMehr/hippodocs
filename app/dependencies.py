from typing import Annotated

from fastapi import Depends, Request
from langchain.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from sqlalchemy.orm import Session

from app.configs import get_settings
from app.database import get_db
from app.services.agent import RagService
from app.services.embedding import (
    DocumentEmbeddingService,
    DocumentIngestionPipeline,
    ExactMatchDeduplicator,
)
from app.services.factory import get_document_repository
from app.services.repository import DocumentRepository


def _get_llm_provider(request: Request) -> BaseChatModel:
    return request.app.state.llm_provider


def _get_cache_embeddings(request: Request) -> Embeddings:
    return request.app.state.cache_embeddings


def _get_embedding_service(request: Request) -> DocumentEmbeddingService:
    return request.app.state.embedding_service


def _get_ingestion_pipeline(
    request: Request,
    repository: DocumentRepository = Depends(get_document_repository),
) -> DocumentIngestionPipeline:
    return DocumentIngestionPipeline(
        embedding_service=request.app.state.embedding_service,
        repository=repository,
        deduplicator=ExactMatchDeduplicator(repository),
    )


def _get_rag_service(
    request: Request,
    session: Session = Depends(get_db),
) -> RagService:
    return RagService(
        db_session=session,
        embeddings_model=request.app.state.cache_embeddings,
        llm=request.app.state.llm_provider,
        top_k=get_settings().TOP_K,
    )


LLMProviderDep = Annotated[object, Depends(_get_llm_provider)]
CacheEmbeddingsDep = Annotated[object, Depends(_get_cache_embeddings)]
EmbeddingServiceDep = Annotated[
    DocumentEmbeddingService, Depends(_get_embedding_service)
]
IngestionPipelineDep = Annotated[
    DocumentIngestionPipeline, Depends(_get_ingestion_pipeline)
]
RagServiceDep = Annotated[RagService, Depends(_get_rag_service)]
DocumentRepositoryDep = Annotated[DocumentRepository, Depends(get_document_repository)]
SessionDep = Annotated[Session, Depends(get_db)]
