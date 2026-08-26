from fastapi import APIRouter, Request

from app.limiter import limiter
from app.schemas import AskChatSchema, ChatResponseSchema

router = APIRouter(prefix="/workspaces/{workspace_id}/ask", tags=["ask"])


@router.post("/", response_model=ChatResponseSchema)
@limiter.limit("30/minute")
def ask_question(request: Request, workspace_id: int, payload: AskChatSchema):
    answer = request.app.state.rag_service.query(workspace_id, payload.question)
    return ChatResponseSchema(content=answer)
