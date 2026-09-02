from fastapi import APIRouter, Query, Request, status

from app.api.dependencies import workspace_service_dep
from app.exceptions import WorkspaceNotFound
from app.limiter import limiter
from app.schemas import (
    WorkspaceCreateSchema,
    WorkspaceListSchema,
    WorkspaceSchema,
)
from app.services.rag_service import WorkspaceService

router = APIRouter(prefix="/workspaces", tags=["workspaces"])


@router.post("/", status_code=status.HTTP_201_CREATED, response_model=WorkspaceSchema)
@limiter.limit("60/minute")
def add_workspace(
    request: Request,
    payload: WorkspaceCreateSchema,
    workspace_service: WorkspaceService = workspace_service_dep,
):
    workspace_id = workspace_service.new_workspace(name=payload.name)
    return WorkspaceSchema(name=payload.name, workspace_id=workspace_id)


@router.get("/", response_model=WorkspaceListSchema)
@limiter.limit("120/minute")
def list_workspaces(
    request: Request,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    workspace_service: WorkspaceService = workspace_service_dep,
):
    workspaces, total = workspace_service.get_workspaces(skip, limit)
    ws_schemas = [
        WorkspaceSchema(workspace_id=ws.workspace_id, name=ws.name)  # type: ignore[attr-defined]
        for ws in workspaces
    ]
    return WorkspaceListSchema(workspaces=ws_schemas, total=total)


@router.get("/{workspace_id}", response_model=WorkspaceSchema)
@limiter.limit("120/minute")
def get_workspace(
    request: Request,
    workspace_id: int,
    workspace_service: WorkspaceService = workspace_service_dep,
):
    workspace = workspace_service.get_workspace(workspace_id)
    if workspace is None:
        raise WorkspaceNotFound(workspace_id)
    return WorkspaceSchema(
        workspace_id=workspace.workspace_id,  # type: ignore[attr-defined]
        name=workspace.name,
    )


@router.delete("/{workspace_id}", status_code=status.HTTP_204_NO_CONTENT)
@limiter.limit("60/minute")
def delete_workspace(
    request: Request,
    workspace_id: int,
    workspace_service: WorkspaceService = workspace_service_dep,
) -> None:
    workspace_service.delete_workspace(workspace_id)
