import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Self


class EmptyContentError(ValueError):
    pass


class EmptyVectorError(ValueError):
    pass


@dataclass(frozen=True)
class Content:
    value: str

    def __post_init__(self) -> None:
        if not self.value.strip():
            raise EmptyContentError("Chunk content cannot be empty or whitespace.")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.value.encode("utf-8")).hexdigest()


@dataclass
class Workspace:
    name: str = field(default="Untitled Workspace")

    @property
    def attributes(self) -> tuple[str, ...]:
        return ("name",)


class DocumentStatus(str, Enum):
    PENDING = "pending"
    CHUNKING_FAILED = "chunking_failed"
    CHUNKED = "chunked"
    EMBED_FAILED = "embed_failed"
    DONE = "done"


@dataclass
class Document:
    content: Content
    workspace: Workspace
    title: str | None = None
    status: DocumentStatus = DocumentStatus.PENDING

    def is_done(self) -> bool:
        return self.status == DocumentStatus.DONE

    @property
    def text(self) -> str:
        return self.content.value


@dataclass(frozen=True)
class Embedding:
    vector: tuple[float, ...]
    model_id: str | None = None

    def __post_init__(self) -> None:
        if self.vector is not None and len(self.vector) == 0:
            raise EmptyVectorError("Embedding vector cannot be empty.")

        if self.model_id is not None and not self.model_id.strip():
            raise ValueError("model_id cannot be an empty string or whitespace.")

    @property
    def ndim(self) -> int:
        return len(self.vector)


@dataclass
class Chunk:
    document_id: int
    workspace_id: int
    content: Content
    embedding_vector: list[float] | None = None
    embedding_model_id: str | None = None

    def has_embedding(self) -> bool:
        return self.embedding is not None

    @property
    def embedding(self) -> Embedding | None:
        if self.embedding_vector is None:
            return None
        return Embedding(
            vector=tuple(self.embedding_vector), model_id=self.embedding_model_id
        )

    @embedding.setter
    def embedding(self, emb: Embedding | None) -> None:
        if emb is None:
            self.embedding_vector = None
            self.embedding_model_id = None
        else:
            self.embedding_vector = list(emb.vector)
            self.embedding_model_id = emb.model_id

    def add_embedding(self, embedding: Embedding) -> Self:
        self.embedding_vector = list(embedding.vector)
        self.embedding_model_id = embedding.model_id
        return self

    @property
    def ndim(self) -> int:
        return len(self.embedding_vector) if self.embedding_vector is not None else 0

    @property
    def text(self) -> str:
        return self.content.value
