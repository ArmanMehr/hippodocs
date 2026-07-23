from typing import Annotated

from fastapi import Depends, Request
from langchain.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.services.agent import RagService
from app.services.embedding import DocumentEmbeddingService, DocumentIngestionPipeline
from app.services.factory import get_document_repository
from app.services.repository import DocumentRepository


def _get_llm_provider(request: Request) -> BaseChatModel:
    return request.app.state.llm_provider


def _get_cache_embeddings(request: Request) -> Embeddings:
    return request.app.state.cache_embeddings


def _get_embedding_service(request: Request) -> DocumentEmbeddingService:
    return request.app.state.embedding_service


def _get_ingestion_pipeline(request: Request) -> DocumentIngestionPipeline:
    return request.app.state.ingestion_pipeline


def _get_rag_service(request: Request) -> RagService:
    return request.app.state.rag_service


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
