from fastapi import Depends
from langfuse import Langfuse

from app.observability import get_langfuse_client
from app.services.factory import (
    create_file_reader,
    create_ingestion_service,
    create_input_validator,
    create_output_validator,
    create_rag_service,
    create_workspace_service,
)
from app.services.rag_service import (
    DocumentIngestionService,
    FileReaderService,
    InputValidatorService,
    OutputValidatorService,
    RagService,
    WorkspaceService,
)


def get_workspace_service() -> WorkspaceService:
    return create_workspace_service()


def get_ingestion_service() -> DocumentIngestionService:
    return create_ingestion_service()


def get_rag_service() -> RagService:
    return create_rag_service()


def get_file_reader() -> FileReaderService:
    return create_file_reader()


def get_input_validator() -> InputValidatorService:
    return create_input_validator()


def get_output_validator() -> OutputValidatorService:
    return create_output_validator()


def get_lengfuse() -> Langfuse | None:
    return get_langfuse_client()


workspace_service_dep = Depends(get_workspace_service)
ingestion_service_dep = Depends(get_ingestion_service)
rag_service_dep = Depends(get_rag_service)
file_reader_dep = Depends(get_file_reader)
input_validator_dep = Depends(get_input_validator)
output_validator_dep = Depends(get_output_validator)
langfuse_client_dep = Depends(get_lengfuse)
