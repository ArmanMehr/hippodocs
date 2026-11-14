from collections.abc import Sequence
from typing import Protocol

from app.domain.models import Content, Embedding
from app.domain.views import RetrievedChunkView


class FileReader(Protocol):
    def validate(self, content: bytes, header: bytes) -> None: ...
    def read(self, content: bytes) -> str: ...


class TextSplitter(Protocol):
    def split_text(self, text: str) -> list[Content]: ...


class TextEmbedder(Protocol):
    model_id: str

    def embed_texts(self, texts: Sequence[str]) -> list[Embedding]: ...
    def embed_query(self, text: str) -> Embedding: ...


class LLMChat(Protocol):
    model_id: str
    system_prompt: str

    def invoke(self, query: str) -> str: ...


class PromptTemplate(Protocol):
    def format(self, **kwargs: str) -> str: ...


class InputSanitizer(Protocol):
    def sanitize(self, text: str) -> str: ...


class PIIRedactor(Protocol):
    def redact(self, text: str) -> str: ...


class OutputValidator(Protocol):
    def validate(self, answer: str) -> str: ...


class Retriever(Protocol):
    def retrieve_in_document(
        self, document_id: int, query_text: str, top_k: int
    ) -> list[RetrievedChunkView]: ...

    def retrieve_in_workspace(
        self, workspace_id: int, query_text: str, top_k: int
    ) -> list[RetrievedChunkView]: ...
