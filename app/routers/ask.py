from fastapi import APIRouter, HTTPException, Request, status

from app.limiter import limiter
from app.schemas import AskChatSchema, ChatResponseSchema
from app.services.rag_service import UnknownWorkspaceError

router = APIRouter(prefix="/ask", tags=["ask"])


@router.post("/", response_model=ChatResponseSchema)
@limiter.limit("30/minute")
def ask_question(request: Request, payload: AskChatSchema):
    try:
        answer = request.app.state.rag_service.query(
            payload.workspace_id, payload.question
        )
    except UnknownWorkspaceError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found"
        )

    return ChatResponseSchema(content=answer)
