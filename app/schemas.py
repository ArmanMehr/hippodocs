from pydantic import BaseModel, Field, field_validator


class DocumentSchema(BaseModel):
    id: int
    content: str


class DocumentListSchema(BaseModel):
    documents: list[DocumentSchema]
    total: int


class DocumentCreateSchema(BaseModel):
    text: str = Field(min_length=1, max_length=100_000)

    @field_validator("text")
    @classmethod
    def strip_text(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text must not be empty or whitespace")
        return v


class AskChatSchema(BaseModel):
    question: str = Field(min_length=1, max_length=5_000)


class ChatResponseSchema(BaseModel):
    content: str
