from fastapi import Depends

from app.adapters.file_reader import FileReaderRegistry
from app.services.factory import (
    create_file_readers,
    create_ingestion_service,
    create_rag_service,
    create_workspace_service,
)
from app.services.rag_service import (
    DocumentIngestionService,
    RagService,
    WorkspaceService,
)


def get_workspace_service() -> WorkspaceService:
    return create_workspace_service()


def get_ingestion_service() -> DocumentIngestionService:
    return create_ingestion_service()


def get_rag_service() -> RagService:
    return create_rag_service()


def get_file_readers() -> FileReaderRegistry:
    return create_file_readers()


workspace_service_dep = Depends(get_workspace_service)
ingestion_service_dep = Depends(get_ingestion_service)
rag_service_dep = Depends(get_rag_service)
file_readers_dep = Depends(get_file_readers)
