from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.configs import get_settings


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    content: Mapped[str] = mapped_column(index=True)
    embedding: Mapped[list[float]] = mapped_column(
        Vector(get_settings().DIMENSIONS), index=True
    )

    def __repr__(self) -> str:
        return f"content: {self.content}, embedding: {self.embedding[:3]}"
