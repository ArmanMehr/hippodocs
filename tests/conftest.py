from collections.abc import Iterable, Sequence
from types import SimpleNamespace
from typing import override

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import TextSplitter
from sqlalchemy.orm import Session

from app.adapters.repository import DocumentRepository
from app.models import Document as DocumentModel
from app.services.embedding import ExactMatchDeduplicator


class FakeSplitter(TextSplitter):
    def __init__(self, split_docs: Sequence[Document]) -> None:
        super().__init__()
        self.split_docs: list[Document] = list(split_docs)
        self.seen: list[Document] = []

    @override
    def split_documents(self, documents: Iterable[Document]) -> list[Document]:
        self.seen = list(documents)
        return list(self.split_docs)

    @override
    def split_text(self, text: str) -> list[str]:
        return [text]


class FakeEmbeddingModel(Embeddings):
    def __init__(self, vectors: Sequence[Sequence[float]] | None = None) -> None:
        self.vectors: list[list[float]] = [list(vector) for vector in (vectors or [])]
        self.queries: list[str] = []
        self.texts: list[str] = []

    @override
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.texts = list(texts)
        return list(self.vectors)

    @override
    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [0.1, 0.2, 0.3]


class FakeDeduplicator(ExactMatchDeduplicator):
    def __init__(self, kept: Sequence[Document], repo_mock: object = None) -> None:
        super().__init__(repository=repo_mock)  # type: ignore[arg-type]
        self.kept: list[Document] = list(kept)
        self.seen: list[Document] = []

    @override
    def filter_new(self, documents: Sequence[Document]) -> list[Document]:
        self.seen = list(documents)
        return list(self.kept)


class FakeRepository(DocumentRepository):
    def __init__(self, session_mock: object = None) -> None:
        super().__init__(session=session_mock)  # type: ignore[arg-type]
        self.saved: list[DocumentModel] | None = None

    @override
    def save_all(self, documents: Sequence[DocumentModel]) -> None:
        self.saved = list(documents)


class FakeSession(Session):
    def __init__(self, rows: Sequence[str]) -> None:
        self.rows = list(rows)
        self.last_stmt: object | None = None

    @override
    def scalars(self, stmt: object) -> SimpleNamespace:  # type: ignore[override]
        self.last_stmt = stmt
        return SimpleNamespace(all=lambda: list(self.rows))


@pytest.fixture
def fake_splitter() -> type[FakeSplitter]:
    return FakeSplitter


@pytest.fixture
def fake_embeddings() -> type[FakeEmbeddingModel]:
    return FakeEmbeddingModel


@pytest.fixture
def fake_deduplicator() -> type[FakeDeduplicator]:
    return FakeDeduplicator


@pytest.fixture
def fake_repository() -> type[FakeRepository]:
    return FakeRepository


@pytest.fixture
def fake_session() -> type[FakeSession]:
    return FakeSession
