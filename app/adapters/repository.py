from collections.abc import Sequence
from typing import Protocol

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Document


class DocumentRepository(Protocol):
    _session: Session

    def get_existing_contents(self, contents: Sequence[str]) -> set[str]: ...

    def save_all(self, documents: Sequence[Document]) -> None: ...

    def list_all(self, skip: int, limit: int) -> tuple[Sequence[Document], int]: ...

    def get_by_id(self, document_id: int) -> Document | None: ...

    def find_similar(self, query_vector: list[float], top_k: int) -> Sequence[str]: ...

    def delete(self, document_id: int) -> bool: ...


class SQLAlchemyDocumentRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get_existing_contents(self, contents: Sequence[str]) -> set[str]:
        if not contents:
            return set()
        lower_texts = [c.lower() for c in contents]
        stmt = select(Document.content).where(
            func.lower(Document.content).in_(lower_texts)
        )
        existing_rows = self._session.execute(stmt).scalars().all()
        return {content.lower() for content in existing_rows}

    def save_all(self, documents: Sequence[Document]) -> None:
        if not documents:
            return
        self._session.add_all(documents)
        self._session.commit()

    def list_all(self, skip: int, limit: int) -> tuple[Sequence[Document], int]:
        total = self._session.scalar(select(func.count(Document.id))) or 0
        docs = self._session.scalars(select(Document).offset(skip).limit(limit)).all()
        return docs, total

    def get_by_id(self, document_id: int) -> Document | None:
        return self._session.get(Document, document_id)

    def find_similar(self, query_vector: list[float], top_k: int) -> Sequence[str]:
        stmt = (
            select(Document.content)
            .order_by(Document.embedding.cosine_distance(query_vector))
            .limit(top_k)
        )
        return self._session.scalars(stmt).all()

    def delete(self, document_id: int) -> bool:
        doc = self.get_by_id(document_id)
        if not doc:
            return False
        self._session.delete(doc)
        self._session.commit()
        return True
