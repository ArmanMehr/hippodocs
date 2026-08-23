from collections.abc import Generator, Sequence
from typing import Self

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm.session import Session, sessionmaker

from app.adapters.orm import clear_mappers, metadata, start_mappers
from app.adapters.repository import (
    ChunkRepository,
    DocumentRepository,
    WorkspaceRepository,
)

# Add these imports at the top
from app.domain.models import Chunk, Content, Document, Embedding, Workspace


@pytest.fixture
def in_memory_db() -> Engine:
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        conn.execute(text("PRAGMA foreign_keys = ON"))
    metadata.create_all(engine)
    return engine


@pytest.fixture
def session_factory(in_memory_db: Engine) -> Generator[sessionmaker[Session]]:
    start_mappers()
    yield sessionmaker(bind=in_memory_db)
    clear_mappers()


@pytest.fixture
def session(session_factory: sessionmaker[Session]) -> Session:
    return session_factory()


class FakeWorkspaceRepository:
    def __init__(self):
        self._workspaces: dict[int, Workspace] = {}
        self._counter = 1

    def add(self, workspace: Workspace) -> None:
        workspace.workspace_id = self._counter  # type: ignore
        self._workspaces[self._counter] = workspace
        self._counter += 1

    def get(self, workspace_id: int) -> Workspace | None:
        return self._workspaces.get(workspace_id)

    def get_all(self, skip: int, limit: int) -> tuple[list[Workspace], int]:
        all_ws = list(self._workspaces.values())
        return all_ws[skip : skip + limit], len(all_ws)

    def update(self, workspace_id: int, new_workspace: Workspace) -> None:
        ws = self.get(workspace_id)
        if ws is not None:
            ws.name = new_workspace.name

    def delete(self, workspace_id: int) -> None:
        self._workspaces.pop(workspace_id, None)


class FakeDocumentRepository:
    def __init__(self):
        self._documents: dict[int, Document] = {}
        self._counter = 1

    def add(self, document: Document) -> None:
        document.document_id = self._counter  # type: ignore
        self._documents[self._counter] = document
        self._counter += 1

    def get(self, document_id: int) -> Document | None:
        return self._documents.get(document_id)

    def list_by_workspace(self, workspace_id: int) -> Sequence[Document]:
        return [
            doc
            for doc in self._documents.values()
            if doc.workspace and getattr(doc.workspace, "workspace_id", None) == workspace_id
        ]

    def list_unpreprocessed_by_workspace(self, workspace_id: int) -> Sequence[Document]:
        return [
            doc
            for doc in self.list_by_workspace(workspace_id)
            if not doc.is_preprocessed
        ]

    def delete(self, document_id: int) -> None:
        self._documents.pop(document_id, None)


class FakeChunkRepository:
    def __init__(self):
        self._chunks: dict[int, Chunk] = {}
        self._counter = 1

    def save_all(self, chunks: Sequence[Chunk]) -> None:
        for chunk in chunks:
            chunk.chunk_id = self._counter  # type: ignore
            self._chunks[self._counter] = chunk
            self._counter += 1

    def get(self, chunk_id: int) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def find_similar_in_document(
        self, document_id: int, query_vector: Sequence[float], top_k: int
    ) -> Sequence[Chunk]:
        _ = query_vector
        chunks = [
            c
            for c in self._chunks.values()
            if c.document_id == document_id and c.embedding_vector is not None
        ]
        return chunks[:top_k]

    def find_similar_in_workspace(
        self, workspace_id: int, query_vector: Sequence[float], top_k: int
    ) -> Sequence[Chunk]:
        _ = workspace_id, query_vector
        chunks = [c for c in self._chunks.values() if c.embedding_vector is not None]
        return chunks[:top_k]


class FakeUnitOfWork:
    workspaces: WorkspaceRepository
    documents: DocumentRepository
    chunks: ChunkRepository

    def __init__(self):
        self.workspaces = FakeWorkspaceRepository()
        self.documents = FakeDocumentRepository()
        self.chunks = FakeChunkRepository()
        self.committed = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def commit(self) -> None:
        self.committed = True

    def flush(self) -> None:
        pass

    def rollback(self) -> None:
        pass


class FakeSplitter:
    def split(self, text: str) -> list[Content]:
        return [
            Content(value=chunk.strip()) for chunk in text.split(".") if chunk.strip()
        ]


class FakeEmbedder:
    def embed(self, text: str) -> Embedding:
        _ = text
        return Embedding(vector=tuple([0.1] * 10), model_id="test-llama2")


class LLMChat:
    def __init__(self, model_id: str = ""):
        self.model_id = model_id

    def invoke(self, query: str) -> str:
        _ = query
        return ""


class FakePromptTemplate:
    def format_prompt(self, question: str, context: list[str]) -> str:
        return ("Question: {question}\nContext: {context}\nAnswer:").format(
            question=question,
            context="\n".join(context),
        )
