from fastapi import APIRouter, Depends, Request

from app.limiter import limiter
from app.schemas import AskChatSchema, ChatResponseSchema
from app.services.agent import RagService
from app.services.factory import get_rag_service

router = APIRouter(prefix="/ask", tags=["ask"])


@router.post("/", response_model=ChatResponseSchema)
@limiter.limit("30/minute")
def ask_question(
    request: Request,
    payload: AskChatSchema,
    rag_service: RagService = Depends(get_rag_service),
):
    answer = rag_service.query(payload.question)
    return ChatResponseSchema(content=answer)
