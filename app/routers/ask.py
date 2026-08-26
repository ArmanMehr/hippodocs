from fastapi import APIRouter, Request

from app.limiter import limiter
from app.schemas import AskChatSchema, ChatResponseSchema

router = APIRouter(prefix="/ask", tags=["ask"])


@router.post("/", response_model=ChatResponseSchema)
@limiter.limit("30/minute")
def ask_question(request: Request, payload: AskChatSchema):
    answer = request.app.state.rag_service.query(
        payload.workspace_id, payload.question
    )
    return ChatResponseSchema(content=answer)
