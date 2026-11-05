from fastapi import APIRouter, Request

from app.api.dependencies import (
    input_validator_dep,
    output_validator_dep,
    rag_service_dep,
)
from app.limiter import limiter
from app.observability import observe
from app.schemas import AskChatSchema, ChatResponseSchema
from app.services.rag_service import (
    InputValidatorService,
    OutputValidatorService,
    RagService,
)

router = APIRouter(prefix="/workspaces/{workspace_id}/ask", tags=["ask"])


@router.post("/", response_model=ChatResponseSchema)
@limiter.limit("30/minute")
@observe(
    name="ask-question", as_type="chain", capture_input=False, capture_output=False
)
def ask_question(
    request: Request,
    workspace_id: int,
    payload: AskChatSchema,
    input_validator: InputValidatorService = input_validator_dep,
    output_validator: OutputValidatorService = output_validator_dep,
    rag_service: RagService = rag_service_dep,
) -> ChatResponseSchema:
    question_val = input_validator.validate(payload.question)
    answer = rag_service.query(workspace_id, question_val)
    answer_val = output_validator.validate(answer)
    return ChatResponseSchema(content=answer_val)
