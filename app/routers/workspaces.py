from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)

from app.adapters.loader import load_pdf
from app.configs import get_settings
from app.limiter import limiter
from app.schemas import (
    AddDocumentResponseSchema,
    WorkspaceCreateSchema,
    WorkspaceListSchema,
    WorkspaceSchema,
)
from app.services.rag_service import UnknownWorkspaceError

router = APIRouter(prefix="/workspaces", tags=["workspaces"])


@router.post("/", status_code=status.HTTP_201_CREATED, response_model=WorkspaceSchema)
@limiter.limit("60/minute")
def add_workspace(
    request: Request,
    payload: WorkspaceCreateSchema,
):
    workspace_id = request.app.state.workspace_service.new_workspace(name=payload.name)
    return WorkspaceSchema(
        name=payload.name,
        workspace_id=workspace_id,
    )


@router.get("/", response_model=WorkspaceListSchema)
@limiter.limit("120/minute")
def list_workspaces(
    request: Request,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
):
    workspaces, total = request.app.state.workspace_service.get_workspaces(skip, limit)
    return WorkspaceListSchema(
        workspaces=[
            WorkspaceSchema(workspace_id=ws.workspace_id, name=ws.name)
            for ws in workspaces
        ],
        total=total,
    )


@router.get("/{workspace_id}", response_model=WorkspaceSchema)
@limiter.limit("120/minute")
def get_workspace(
    request: Request,
    workspace_id: int,
):
    try:
        workspace = request.app.state.workspace_service.get_workspace(workspace_id)
    except UnknownWorkspaceError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found"
        )
    return WorkspaceSchema(workspace_id=workspace.workspace_id, name=workspace.name)


@router.delete("/{workspace_id}", status_code=status.HTTP_204_NO_CONTENT)
@limiter.limit("60/minute")
def delete_workspace(
    request: Request,
    workspace_id: int,
) -> None:
    try:
        request.app.state.workspace_service.delete_workspace(workspace_id)
    except UnknownWorkspaceError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found"
        )


@router.post(
    "/upload-document",
    status_code=status.HTTP_201_CREATED,
    response_model=AddDocumentResponseSchema,
)
@limiter.limit("10/minute")
def upload_document(
    request: Request,
    workspace_id: int = Form(...),
    file: UploadFile = File(...),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported",
        )

    if file.size is not None and file.size > get_settings().MAX_FILESIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File too large (max 10MB)",
        )

    content = file.file.read()
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file"
        )

    pdf_document = load_pdf(content)
    if not pdf_document:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No text could be extracted from this PDF",
        )

    title = file.filename.rsplit(".", 1)[0] if file.filename else None

    try:
        document_id = request.app.state.workspace_service.new_document(
            workspace_id, title, pdf_document
        )
    except UnknownWorkspaceError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace with workspace id {workspace_id} not found.",
        )

    request.app.state.ingestion_service.ingest_document(document_id=document_id)
    return AddDocumentResponseSchema(
        document_id=document_id, title=title, text=pdf_document
    )
