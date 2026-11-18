import pytest

from app.domain.models import Chunk
from app.exceptions import EmptyContent
from app.services.rag_service import (
    DocumentIngestionService,
    WorkspaceNotFound,
)
from app.services.uow import UnitOfWork
from tests.conftest import (
    FakeEmbedder,
    FakeInputValidator,
    FakeSplitter,
    FakeUnitOfWork,
    seed_workspace,
)


def get_ingestion_service(uow: UnitOfWork) -> DocumentIngestionService:
    return DocumentIngestionService(
        uow=uow,
        splitter=FakeSplitter(),
        embedder=FakeEmbedder(),
        document_validator=FakeInputValidator(),
    )


def get_chunks(uow: UnitOfWork, document_id: int) -> list[Chunk]:
    if isinstance(uow, FakeUnitOfWork):
        return [
            c
            for c in uow.chunks._get_all().values()  # type: ignore
            if c.document_id == document_id and c.embedding_vector is not None
        ]
    return []


def test_ingest_unknown_workspace_raises_error(uow: UnitOfWork):
    with pytest.raises(WorkspaceNotFound):
        get_ingestion_service(uow).add_document("Text", workspace_id=999, title="Title")


def test_retry_ingestion_with_no_documents_is_noop(uow: UnitOfWork):
    workspace_id = seed_workspace(uow)

    get_ingestion_service(uow).retry_ingestion_in_workspace(workspace_id)

    if isinstance(uow, FakeUnitOfWork):
        assert not [
            c
            for c in uow.chunks._get_all().values()  # type: ignore
            if c.workspace_id == workspace_id and c.embedding_vector is not None
        ]


def test_add_document_reuses_existing_document_by_content(uow: UnitOfWork):
    workspace_id = seed_workspace(uow)
    service = get_ingestion_service(uow)

    first_id, _, _ = service.add_document("Text1. Text2", workspace_id, "First")
    second_id, _, _ = service.add_document("Text1. Text2", workspace_id, "Second")

    assert first_id == second_id
    assert len(uow.documents._documents) == 1  # type: ignore[attr-defined]


def test_add_document_returns_id_title_and_validated_content(uow: UnitOfWork):
    workspace_id = seed_workspace(uow)

    document_id, title, content = get_ingestion_service(uow).add_document(
        "Text1. Text2", workspace_id, "Title"
    )

    assert document_id == 1
    assert title == "Title"
    assert content == "Text1. Text2"
    doc = uow.documents.get(document_id)
    assert doc is not None
    assert doc.status.value == "done"


def test_saved_chunks_after_splitting(uow: UnitOfWork):
    workspace_id = seed_workspace(uow)
    document_id, _, _ = get_ingestion_service(uow).add_document(
        "Text1. Text2", workspace_id, "Title"
    )
    chunks = get_chunks(uow, document_id)

    assert [chunk.content.value for chunk in chunks] == ["Text1", "Text2"]
    for chunk in chunks:
        assert chunk.has_embedding()
        assert chunk.embedding_vector == [0.1, 0.1]


def test_empty_document_raises_empty_content_exception(uow: UnitOfWork):
    workspace_id = seed_workspace(uow)

    with pytest.raises(EmptyContent):
        get_ingestion_service(uow).add_document(
            text="  ", workspace_id=workspace_id, title="Title"
        )
