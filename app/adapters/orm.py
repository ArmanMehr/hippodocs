from typing import Any, override

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    Column,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    TypeDecorator,
    UniqueConstraint,
)
from sqlalchemy.orm import composite, registry, relationship

from app.configs import get_settings
from app.domain.models import Chunk, Content, Document, Embedding, Workspace


class EmbeddingVector(TypeDecorator[Vector]):
    impl = Vector(get_settings().DIMENSIONS)
    cache_ok = True

    @override
    def process_bind_param(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None
        if not isinstance(value, list):
            return list(value)
        return value

    @override
    def process_result_value(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None
        if not isinstance(value, list):
            return list(value)
        return value


metadata = MetaData()
mapper_registry = registry(metadata=metadata)


class ContentType(TypeDecorator[str]):
    impl = Text
    cache_ok = True

    @override
    def process_bind_param(self, value: Any, dialect: Any) -> Any:
        if isinstance(value, Content):
            return value.value
        return value

    @override
    def process_result_value(self, value: Any, dialect: Any) -> Any:
        if value is not None:
            return Content(value)
        return value


workspaces_table = Table(
    "workspaces",
    mapper_registry.metadata,
    Column("workspace_id", Integer, primary_key=True, autoincrement=True),
    Column("name", String(255), nullable=True),
)

documents_table = Table(
    "documents",
    mapper_registry.metadata,
    Column("document_id", Integer, primary_key=True, autoincrement=True),
    Column("content", ContentType, nullable=False),
    Column(
        "workspace_id",
        Integer,
        ForeignKey("workspaces.workspace_id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("title", String(255), nullable=True),
    Column("is_preprocessed", Boolean, nullable=False, default=False),
    Index("ix_documents_workspace_preprocessed", "workspace_id", "is_preprocessed"),
)

chunks_table = Table(
    "chunks",
    mapper_registry.metadata,
    Column("chunk_id", Integer, primary_key=True, autoincrement=True),
    Column(
        "document_id",
        Integer,
        ForeignKey("documents.document_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    ),
    Column("content", ContentType, nullable=False),
    Column("embedding_vector", EmbeddingVector(), nullable=True),
    Column("embedding_model_id", String(255), nullable=True),
    UniqueConstraint("document_id", "content", name="uq_document_content"),
)


def start_mappers() -> None:
    mapper_registry.map_imperatively(Workspace, workspaces_table)
    mapper_registry.map_imperatively(
        Document,
        documents_table,
        properties={
            "_is_preprocessed": documents_table.c.is_preprocessed,
            "workspace": relationship(Workspace, lazy="select"),
        },
    )
    mapper_registry.map_imperatively(
        Chunk,
        chunks_table,
        properties={
            "embedding": composite(
                Embedding,
                chunks_table.c.embedding_vector,
                chunks_table.c.embedding_model_id,
            ),
        },
    )


def clear_mappers():
    mapper_registry.dispose()
