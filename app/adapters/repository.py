from collections.abc import Sequence
from typing import Protocol

from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from app.adapters.orm import chunks_table, documents_table, workspaces_table
from app.domain.models import Chunk, Document, Workspace


class UnknownWorkspaceError(KeyError):
    pass


class UnknownDocumentError(KeyError):
    pass


class WorkspaceRepository(Protocol):
    def add(self, workspace: Workspace) -> None: ...

    def get(self, workspace_id: int) -> Workspace: ...

    def get_all(self, skip: int, limit: int) -> tuple[list[Workspace], int]: ...

    def update(self, workspace_id: int, new_workspace: Workspace) -> None: ...

    def delete(self, workspace_id: int) -> None: ...


class DocumentRepository(Protocol):
    def add(self, document: Document) -> None: ...

    def get(self, document_id: int) -> Document: ...

    def list_by_workspace(self, workspace_id: int) -> Sequence[Document]: ...

    def list_unpreprocessed_by_workspace(
        self, workspace_id: int
    ) -> Sequence[Document]: ...

    def delete(self, document_id: int) -> None: ...


class ChunkRepository(Protocol):
    def save_all(self, chunks: Sequence[Chunk]) -> None: ...

    def get(self, chunk_id: int) -> Chunk: ...

    def find_similar_in_workspace(
        self, workspace_id: int, query_vector: Sequence[float], top_k: int
    ) -> Sequence[Chunk]: ...

    def find_similar_in_document(
        self, document_id: int, query_vector: Sequence[float], top_k: int
    ) -> Sequence[Chunk]: ...


class SQLAlchemyWorkspaceRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, workspace: Workspace) -> None:
        self._session.add(workspace)

    def get(self, workspace_id: int) -> Workspace:
        stmt = select(Workspace).where(workspaces_table.c.workspace_id == workspace_id)
        ws = self._session.scalar(stmt)
        if ws is None:
            raise UnknownWorkspaceError(workspace_id)
        return ws

    def get_all(self, skip: int, limit: int) -> tuple[list[Workspace], int]:
        stmt = (
            select(Workspace, func.count().over().label("total_count"))
            .offset(skip)
            .limit(limit)
        )

        rows = self._session.execute(stmt).all()
        if not rows:
            return [], 0

        workspaces = [row[0] for row in rows]
        total = rows[0][1]

        return workspaces, total

    def update(self, workspace_id: int, new_workspace: Workspace) -> None:
        ws = self.get(workspace_id)
        for attr in new_workspace.attribures:
            setattr(ws, attr, getattr(new_workspace, attr))

    def delete(self, workspace_id: int) -> None:
        ws = self.get(workspace_id)
        self._session.delete(ws)


class SQLAlchemyDocumentRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, document: Document) -> None:
        self._session.add(document)

    def get(self, document_id: int) -> Document:
        stmt = select(Document).where(documents_table.c.document_id == document_id)
        doc = self._session.scalar(stmt)
        if doc is None:
            raise UnknownDocumentError(document_id)
        return doc

    def list_by_workspace(self, workspace_id: int) -> Sequence[Document]:
        stmt = select(Document).where(documents_table.c.workspace_id == workspace_id)
        return list(self._session.scalars(stmt))

    def list_unpreprocessed_by_workspace(self, workspace_id: int) -> Sequence[Document]:
        stmt = select(Document).where(
            and_(
                documents_table.c.workspace_id == workspace_id,
                documents_table.c.is_preprocessed == False,
            )
        )
        return list(self._session.scalars(stmt))

    def delete(self, document_id: int) -> None:
        doc = self.get(document_id)
        self._session.delete(doc)


class SQLAlchemyChunkRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def save_all(self, chunks: Sequence[Chunk]) -> None:
        self._session.add_all(chunks)

    def get(self, chunk_id: int) -> Chunk:
        stmt = select(Chunk).where(chunks_table.c.chunk_id == chunk_id)
        chunk = self._session.scalar(stmt)
        if chunk is None:
            raise KeyError(chunk_id)
        return chunk

    def find_similar_in_document(
        self, document_id: int, query_vector: Sequence[float], top_k: int
    ) -> Sequence[Chunk]:
        stmt = (
            select(Chunk)
            .where(
                and_(
                    chunks_table.c.document_id == document_id,
                    chunks_table.c.embedding_vector.is_not(None),
                )
            )
            .order_by(chunks_table.c.embedding_vector.cosine_distance(query_vector))
            .limit(top_k)
        )
        return list(self._session.scalars(stmt))

    def find_similar_in_workspace(
        self, workspace_id: int, query_vector: Sequence[float], top_k: int
    ) -> Sequence[Chunk]:
        stmt = (
            select(Chunk)
            .join(
                documents_table,
                chunks_table.c.document_id == documents_table.c.document_id,
            )
            .where(
                and_(
                    documents_table.c.workspace_id == workspace_id,
                    chunks_table.c.embedding_vector.is_not(None),
                )
            )
            .order_by(chunks_table.c.embedding_vector.cosine_distance(query_vector))
            .limit(top_k)
        )
        return list(self._session.scalars(stmt))
