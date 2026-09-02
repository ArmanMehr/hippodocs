from fastapi import APIRouter, Request

from app.api.dependencies import rag_service_dep
from app.limiter import limiter
from app.schemas import AskChatSchema, ChatResponseSchema
from app.services.rag_service import RagService

router = APIRouter(prefix="/workspaces/{workspace_id}/ask", tags=["ask"])


@router.post("/", response_model=ChatResponseSchema)
@limiter.limit("30/minute")
def ask_question(
    request: Request,
    workspace_id: int,
    payload: AskChatSchema,
    rag_service: RagService = rag_service_dep,
):
    answer = rag_service.query(workspace_id, payload.question)
    return ChatResponseSchema(content=answer)
