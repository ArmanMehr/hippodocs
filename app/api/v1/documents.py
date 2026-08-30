from fastapi import APIRouter, File, Path, Query, Request, UploadFile, status

from app.configs import get_settings
from app.exceptions import FileTooLarge, MissingFilename
from app.limiter import limiter
from app.schemas import (
    AddDocumentResponseSchema,
    DocumentListSchema,
    DocumentSchema,
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
):
    documents, total = request.app.state.workspace_service.list_documents_in_workspace(
        workspace_id, skip=skip, limit=limit
    )
    doc_schemas = [
        DocumentSchema(document_id=doc.document_id, title=doc.title)
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
):
    if not file.filename:
        raise MissingFilename()

    if file.size is not None and file.size > get_settings().MAX_FILESIZE:
        raise FileTooLarge()

    extension = file.filename.rsplit(".", 1)[-1].lower()
    reader = request.app.state.file_readers.get(extension)

    content = file.file.read()
    title = file.filename.rsplit(".", 1)[0]

    document_id = request.app.state.ingestion_service.add_document(
        reader=reader,
        file_data=content,
        workspace_id=workspace_id,
        title=title,
    )

    return AddDocumentResponseSchema(document_id=document_id, title=title, text=content)


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
@limiter.limit("60/minute")
def delete_document(
    request: Request,
    workspace_id: int = Path(...),
    document_id: int = Path(...),
) -> None:
    request.app.state.workspace_service.delete_document(workspace_id, document_id)
