from collections.abc import Sequence
from typing import Protocol

from app.domain.models import Chunk, Content, Embedding
from app.domain.views import RetrievedChunkView


class FileReader(Protocol):
    def validate(self, content: bytes, header: bytes) -> None: ...
    def read(self, content: bytes) -> str: ...


class TextSplitter(Protocol):
    def split_text(
        self, text: str, document_id: int, workspace_id: int
    ) -> list[Chunk]: ...


class TextEmbedder(Protocol):
    model_id: str

    def embed_texts(self, texts: Sequence[str]) -> list[Embedding]: ...
    def embed_query(self, text: str) -> Embedding: ...


class LLMChat(Protocol):
    model_id: str
    system_prompt: str

    def invoke(self, query: Content) -> Content: ...


class PromptTemplate(Protocol):
    def format(self, **kwargs: str) -> Content: ...


class InputValidator(Protocol):
    def validate(self, content: Content) -> Content: ...


class OutputValidator(Protocol):
    def validate(self, content: Content) -> Content: ...


class Retriever(Protocol):
    def retrieve_in_document(
        self, document_id: int, query: Content, top_k: int
    ) -> list[RetrievedChunkView]: ...

    def retrieve_in_workspace(
        self, workspace_id: int, query: Content, top_k: int
    ) -> list[RetrievedChunkView]: ...
