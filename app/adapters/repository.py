from collections.abc import Sequence
from typing import Protocol

from sqlalchemy import and_, func, insert, select, update
from sqlalchemy.orm import Session
from sqlalchemy.sql.expression import bindparam

from app.adapters.orm import (
    chunks_table,
    document_hashes_table,
    documents_table,
    workspaces_table,
)
from app.domain.models import Chunk, Document, DocumentStatus, Workspace


class WorkspaceRepository(Protocol):
    def add(self, workspace: Workspace) -> None: ...

    def get(self, workspace_id: int) -> Workspace | None: ...

    def get_all(self, skip: int, limit: int) -> tuple[list[Workspace], int]: ...

    def update(self, workspace_id: int, new_workspace: Workspace) -> None: ...

    def delete(self, workspace_id: int) -> None: ...


class DocumentRepository(Protocol):
    def add(self, document: Document) -> None: ...

    def add_content_hash(
        self, workspace_id: int, document_id: int, content_hash: str
    ) -> None: ...

    def get(self, document_id: int) -> Document | None: ...

    def list_by_workspace(
        self, workspace_id: int, skip: int, limit: int
    ) -> tuple[list[Document], int]: ...

    def list_undone_by_workspace(self, workspace_id: int) -> list[Document]: ...

    def get_identical_document_in_workspace(
        self, workspace_id: int, content_hash: str
    ) -> Document | None: ...

    def update_status(self, document_id: int, new_status: DocumentStatus) -> None: ...

    def delete(self, document_id: int) -> None: ...


class ChunkRepository(Protocol):
    def save_all(self, chunks: Sequence[Chunk]) -> None: ...

    def get(self, chunk_id: int) -> Chunk | None: ...

    def get_by_document(self, document_id: int) -> list[Chunk]: ...

    def update_embeddings(self, chunks: Sequence[Chunk]) -> None: ...


class SQLAlchemyWorkspaceRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, workspace: Workspace) -> None:
        self._session.add(workspace)

    def get(self, workspace_id: int) -> Workspace | None:
        stmt = select(Workspace).where(workspaces_table.c.workspace_id == workspace_id)
        return self._session.scalar(stmt)

    def get_all(self, skip: int, limit: int) -> tuple[list[Workspace], int]:
        stmt = (
            select(Workspace, func.count().over().label("total_count"))
            .offset(skip)
            .limit(limit)
        )

        rows = self._session.execute(stmt).all()
        if not rows:
            return [], 0

        workspaces: list[Workspace] = [row[0] for row in rows]
        total: int = rows[0][1]

        return workspaces, total

    def update(self, workspace_id: int, new_workspace: Workspace) -> None:
        ws = self.get(workspace_id)
        if ws is None:
            return
        for attr in new_workspace.attributes:
            setattr(ws, attr, getattr(new_workspace, attr))

    def delete(self, workspace_id: int) -> None:
        ws = self.get(workspace_id)
        if ws is not None:
            self._session.delete(ws)


class SQLAlchemyDocumentRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, document: Document) -> None:
        self._session.add(document)

    def add_content_hash(
        self, workspace_id: int, document_id: int, content_hash: str
    ) -> None:
        stmt = insert(document_hashes_table).values(
            document_id=document_id,
            workspace_id=workspace_id,
            content_hash=content_hash,
        )
        self._session.execute(stmt)

    def get(self, document_id: int) -> Document | None:
        stmt = select(Document).where(documents_table.c.document_id == document_id)
        return self._session.scalar(stmt)

    def list_by_workspace(
        self, workspace_id: int, skip: int, limit: int
    ) -> tuple[list[Document], int]:
        stmt = (
            select(Document, func.count().over().label("total_count"))
            .where(documents_table.c.workspace_id == workspace_id)
            .offset(skip)
            .limit(limit)
        )
        rows = self._session.execute(stmt).all()
        if not rows:
            return [], 0

        documents: list[Document] = [row[0] for row in rows]
        total: int = rows[0][1]

        return documents, total

    def list_undone_by_workspace(self, workspace_id: int) -> list[Document]:
        stmt = select(Document).where(
            and_(
                documents_table.c.workspace_id == workspace_id,
                documents_table.c.status != DocumentStatus.DONE.value,
            )
        )
        return list(self._session.scalars(stmt).all())

    def get_identical_document_in_workspace(
        self, workspace_id: int, content_hash: str
    ) -> Document | None:
        stmt = (
            select(Document)
            .join(
                document_hashes_table,
                documents_table.c.document_id == document_hashes_table.c.document_id,
            )
            .where(
                and_(
                    document_hashes_table.c.workspace_id == workspace_id,
                    document_hashes_table.c.content_hash == content_hash,
                )
            )
            .limit(1)
        )
        return self._session.scalar(stmt)

    def update_status(self, document_id: int, new_status: DocumentStatus) -> None:
        stmt = (
            update(documents_table)
            .where(documents_table.c.document_id == document_id)
            .values(status=new_status)
        )
        self._session.execute(stmt)

    def delete(self, document_id: int) -> None:
        doc = self.get(document_id)
        if doc is not None:
            self._session.delete(doc)


class SQLAlchemyChunkRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def save_all(self, chunks: Sequence[Chunk]) -> None:
        self._session.add_all(chunks)

    def get(self, chunk_id: int) -> Chunk | None:
        stmt = select(Chunk).where(chunks_table.c.chunk_id == chunk_id)
        return self._session.scalar(stmt)

    def get_by_document(self, document_id: int) -> list[Chunk]:
        stmt = select(Chunk).where(chunks_table.c.document_id == document_id)
        return list(self._session.scalars(stmt).all())

    def update_embeddings(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            return
        stmt = (
            update(chunks_table)
            .where(chunks_table.c.chunk_id == bindparam("p_chunk_id"))
            .values(
                embedding_vector=bindparam("p_embedding_vector"),
                embedding_model_id=bindparam("p_embedding_model_id"),
            )
        )
        params = [
            {
                "p_chunk_id": chunk.chunk_id,  # type: ignore[attr-defined]
                "p_embedding_vector": chunk.embedding_vector,
                "p_embedding_model_id": chunk.embedding_model_id,
            }
            for chunk in chunks
        ]
        self._session.execute(stmt, params)
