from dataclasses import dataclass, field
from typing import Self


class EmptyContentError(ValueError):
    pass


class EmptyVectorError(ValueError):
    pass


@dataclass(frozen=True)
class Content:
    value: str

    def __post_init__(self):
        if not self.value.strip():
            raise EmptyContentError("Chunk content cannot be empty or whitespace.")


@dataclass
class Workspace:
    name: str = field(default="Untitled Workspace")

    @property
    def attribures(self) -> tuple[str, ...]:
        return ("name",)


@dataclass
class Document:
    content: Content
    workspace: Workspace
    title: str | None = None
    _is_preprocessed: bool = False

    @property
    def is_preprocessed(self) -> bool:
        return self._is_preprocessed

    def mark_preprocessed(self) -> None:
        self._is_preprocessed = True


@dataclass(frozen=True)
class Embedding:
    vector: tuple[float, ...]
    model_id: str | None = None

    def __post_init__(self):
        if self.vector is not None and len(self.vector) == 0:
            raise EmptyVectorError("Embedding vector cannot be empty.")

        if self.model_id is not None and not self.model_id.strip():
            raise ValueError("model_id cannot be an empty string or whitespace.")

    @property
    def ndim(self):
        return len(self.vector)


@dataclass
class Chunk:
    document_id: int
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
