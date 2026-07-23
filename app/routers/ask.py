from fastapi import APIRouter, Request

from app.dependencies import RagServiceDep
from app.limiter import limiter
from app.schemas import AskChatSchema, ChatResponseSchema

router = APIRouter(prefix="/ask", tags=["ask"])


@router.post("/", response_model=ChatResponseSchema)
@limiter.limit("30/minute")
def ask_question(
    request: Request,
    payload: AskChatSchema,
    rag_service: RagServiceDep,
):
    answer = rag_service.query(payload.question)
    return ChatResponseSchema(content=answer)
