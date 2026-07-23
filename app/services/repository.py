from collections.abc import Sequence
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Document


class DocumentRepository:
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

    def get_by_id(self, document_id: int) -> Optional[Document]:
        return self._session.get(Document, document_id)

    def delete(self, document_id: int) -> bool:
        doc = self.get_by_id(document_id)
        if not doc:
            return False
        self._session.delete(doc)
        self._session.commit()
        return True
