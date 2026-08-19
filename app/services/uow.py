from typing import Protocol, Self

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.adapters.repository import (
    ChunkRepository,
    DocumentRepository,
    SQLAlchemyChunkRepository,
    SQLAlchemyDocumentRepository,
    SQLAlchemyWorkspaceRepository,
    WorkspaceRepository,
)
from app.configs import get_settings

DEFAULT_SESSION_FACTORY = sessionmaker(
    create_engine(
        get_settings().DATABASE_URL,
        isolation_level="REPEATABLE READ",
    )
)


class UnitOfWork(Protocol):
    workspaces: WorkspaceRepository
    documents: DocumentRepository
    chunks: ChunkRepository

    def __enter__(self) -> Self: ...

    def __exit__(self, *args: object) -> None: ...

    def commit(self) -> None: ...

    def flush(self) -> None: ...

    def rollback(self) -> None: ...


class SQLAlchemyUnitOfWork:
    workspaces: WorkspaceRepository
    documents: DocumentRepository
    chunks: ChunkRepository
    session: Session

    def __init__(
        self, session_factory: sessionmaker[Session] = DEFAULT_SESSION_FACTORY
    ):
        self.session_factory = session_factory

    def __enter__(self) -> Self:
        self.session = self.session_factory()
        self.workspaces = SQLAlchemyWorkspaceRepository(self.session)
        self.documents = SQLAlchemyDocumentRepository(self.session)
        self.chunks = SQLAlchemyChunkRepository(self.session)
        return self

    def __exit__(self, *args: object) -> None:
        self.session.close()

    def commit(self):
        self.session.commit()

    def rollback(self):
        self.session.rollback()

    def flush(self):
        self.session.flush()
