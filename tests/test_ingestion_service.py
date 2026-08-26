import pytest

from app.domain.models import Chunk
from app.services.rag_service import (
    DocumentIngestionService,
    UnknownDocumentError,
    UnknownWorkspaceError,
)
from app.services.uow import UnitOfWork
from tests.conftest import FakeEmbedder, FakeSplitter, seed_document, seed_workspace


def get_ingestion_service(uow: UnitOfWork) -> DocumentIngestionService:
    return DocumentIngestionService(uow=uow, splitter=FakeSplitter(), embedder=FakeEmbedder())


def get_chunks(uow: UnitOfWork, document_id: int) -> list[Chunk]:
    return list(
        uow.chunks.find_similar_in_document(
            document_id=document_id, query_vector=[0.0], top_k=10
        )
    )


@pytest.fixture
def document_id(uow: UnitOfWork) -> int:
    return seed_document(uow, "Text1. Text2")


@pytest.fixture
def ingested_document(document_id: int, uow: UnitOfWork) -> int:
    get_ingestion_service(uow).ingest_document(document_id)
    return document_id


def test_ingest_unknown_document_raises_error(uow: UnitOfWork):
    with pytest.raises(UnknownDocumentError):
        get_ingestion_service(uow).ingest_document(document_id=999)


def test_ingest_unknown_workspace_raises_error(uow: UnitOfWork):
    with pytest.raises(UnknownWorkspaceError):
        get_ingestion_service(uow).ingest_workspace(workspace_id=999)


def test_ingest_workspace_with_no_documents_is_noop(uow: UnitOfWork):
    workspace_id = seed_workspace(uow)

    get_ingestion_service(uow).ingest_workspace(workspace_id)

    assert not uow.chunks.find_similar_in_workspace(
        workspace_id, query_vector=[0.0], top_k=10
    )


def test_no_content_after_splitting(uow: UnitOfWork):
    doc_id = seed_document(uow, ". ")

    get_ingestion_service(uow).ingest_document(doc_id)

    doc = uow.documents.get(doc_id)
    assert doc
    assert doc.is_preprocessed
    assert not get_chunks(uow, doc_id)


def test_marks_document_preprocessed(ingested_document: int, uow: UnitOfWork):
    doc = uow.documents.get(ingested_document)
    assert doc
    assert doc.is_preprocessed


def test_saved_chunks_after_splitting(ingested_document: int, uow: UnitOfWork):
    chunks = get_chunks(uow, ingested_document)

    assert [chunk.content.value for chunk in chunks] == ["Text1", "Text2"]
    for chunk in chunks:
        assert chunk.has_embedding()
        assert chunk.embedding_vector == [0.1, 0.1]
