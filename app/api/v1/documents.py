from fastapi import APIRouter, File, Path, Query, Request, UploadFile, status

from app.api.dependencies import (
    file_reader_dep,
    ingestion_service_dep,
    input_validator_dep,
    workspace_service_dep,
)
from app.configs import get_settings
from app.exceptions import FileTooLarge, MissingFilename
from app.limiter import limiter
from app.schemas import (
    AddDocumentResponseSchema,
    DocumentListSchema,
    DocumentSchema,
)
from app.services.rag_service import (
    DocumentIngestionService,
    FileReaderService,
    InputValidatorService,
    WorkspaceService,
)

router = APIRouter(prefix="/workspaces/{workspace_id}", tags=["documents"])


@router.get(
    "/documents", status_code=status.HTTP_200_OK, response_model=DocumentListSchema
)
@limiter.limit("120/minute")
def get_documents(
    request: Request,
    workspace_id: int = Path(...),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    workspace_service: WorkspaceService = workspace_service_dep,
):
    documents, total = workspace_service.list_documents_in_workspace(
        workspace_id, skip=skip, limit=limit
    )
    doc_schemas = [
        DocumentSchema(document_id=doc.document_id, title=doc.title)  # type: ignore[attr-defined]
        for doc in documents
    ]
    return DocumentListSchema(documents=doc_schemas, total=total)


@router.post(
    "/documents",
    status_code=status.HTTP_201_CREATED,
    response_model=AddDocumentResponseSchema,
)
@limiter.limit("10/minute")
def upload_document(
    request: Request,
    workspace_id: int = Path(...),
    file: UploadFile = File(...),  # noqa: B008
    file_reader: FileReaderService = file_reader_dep,
    input_validator: InputValidatorService = input_validator_dep,
    ingestion_service: DocumentIngestionService = ingestion_service_dep,
):
    if not file.filename:
        raise MissingFilename()

    if file.size is not None and file.size > get_settings().MAX_FILESIZE:
        raise FileTooLarge()

    text = file_reader.read(file.file, file.filename)
    title = file_reader.get_filename_no_ext(file.filename)

    text = input_validator.validate(text)
    title = input_validator.validate(title)

    document_id, title, text = ingestion_service.add_document(
        text=text, workspace_id=workspace_id, title=title
    )

    return AddDocumentResponseSchema(document_id=document_id, title=title, text=text)


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
@limiter.limit("60/minute")
def delete_document(
    request: Request,
    workspace_id: int = Path(...),
    document_id: int = Path(...),
    workspace_service: WorkspaceService = workspace_service_dep,
) -> None:
    workspace_service.delete_document(workspace_id, document_id)
