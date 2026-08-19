from collections.abc import Sequence
from typing import Protocol

from app.domain.models import Content, Embedding


class TextSplitter(Protocol):
    def split_text(self, text: str) -> list[Content]: ...


class TextEmbedder(Protocol):
    model_id: str

    def embed_texts(self, texts: Sequence[str]) -> list[Embedding]: ...
    def embed_query(self, text: str) -> Embedding: ...


class LLMChat(Protocol):
    model_id: str

    def invoke(self, query: str) -> str: ...


class PromptTemplate(Protocol):
    def format(self, **kwargs: str) -> str: ...
