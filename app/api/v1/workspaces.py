from fastapi import APIRouter, File, Path, Query, Request, UploadFile, status

from app.configs import get_settings
from app.exceptions import FileTooLarge, MissingFilename
from app.limiter import limiter
from app.schemas import (
    AddDocumentResponseSchema,
    WorkspaceCreateSchema,
    WorkspaceListSchema,
    WorkspaceSchema,
)

router = APIRouter(prefix="/workspaces", tags=["workspaces"])


@router.post("/", status_code=status.HTTP_201_CREATED, response_model=WorkspaceSchema)
@limiter.limit("60/minute")
def add_workspace(request: Request, payload: WorkspaceCreateSchema):
    workspace_id = request.app.state.workspace_service.new_workspace(name=payload.name)
    return WorkspaceSchema(name=payload.name, workspace_id=workspace_id)


@router.get("/", response_model=WorkspaceListSchema)
@limiter.limit("120/minute")
def list_workspaces(
    request: Request,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
):
    workspaces, total = request.app.state.workspace_service.get_workspaces(skip, limit)
    ws_schemas = [
        WorkspaceSchema(workspace_id=ws.workspace_id, name=ws.name) for ws in workspaces
    ]
    return WorkspaceListSchema(workspaces=ws_schemas, total=total)


@router.get("/{workspace_id}", response_model=WorkspaceSchema)
@limiter.limit("120/minute")
def get_workspace(request: Request, workspace_id: int):
    workspace = request.app.state.workspace_service.get_workspace(workspace_id)
    return WorkspaceSchema(workspace_id=workspace.workspace_id, name=workspace.name)


@router.delete("/{workspace_id}", status_code=status.HTTP_204_NO_CONTENT)
@limiter.limit("60/minute")
def delete_workspace(request: Request, workspace_id: int) -> None:
    request.app.state.workspace_service.delete_workspace(workspace_id)


@router.post(
    "/{workspace_id}/documents",
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
