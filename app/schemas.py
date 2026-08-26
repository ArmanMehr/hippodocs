from pydantic import BaseModel, Field, field_validator


class WorkspaceSchema(BaseModel):
    workspace_id: int
    name: str


class WorkspaceCreateSchema(BaseModel):
    name: str = Field(min_length=1, max_length=255)

    @field_validator("name")
    def validate_name(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Workspace name cannot be empty or whitespace")
        return v


class WorkspaceListSchema(BaseModel):
    workspaces: list[WorkspaceSchema]
    total: int


class AddDocumentSchema(BaseModel):
    workspace_id: int


class AddDocumentResponseSchema(BaseModel):
    document_id: int
    title: str | None = Field(default=None, max_length=255)
    text: str = Field(default_factory=str, min_length=1, max_length=5_000_000)


class AskChatSchema(BaseModel):
    question: str = Field(min_length=1, max_length=5_000)


class ChatResponseSchema(BaseModel):
    content: str
