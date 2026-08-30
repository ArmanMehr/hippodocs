from fastapi import APIRouter, Query, Request, status

from app.limiter import limiter
from app.schemas import (
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
