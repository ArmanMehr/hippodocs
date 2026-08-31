import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.adapters.orm import chunks_table
from app.adapters.repository import (
    ChunkRepository,
    DocumentRepository,
    SQLAlchemyChunkRepository,
    SQLAlchemyDocumentRepository,
    SQLAlchemyWorkspaceRepository,
    WorkspaceRepository,
)
from app.domain.models import Chunk, Content, Document, Embedding, Workspace


def add_workspace(
    workspace_repo: WorkspaceRepository, session: Session, name: str = "ws"
) -> Workspace:
    ws = Workspace(name=name)
    workspace_repo.add(ws)
    session.commit()
    return ws


def add_document(
    document_repo: DocumentRepository,
    session: Session,
    workspace: Workspace,
    title: str = "doc",
) -> Document:
    doc = Document(content=Content("text"), workspace=workspace, title=title)
    document_repo.add(doc)
    session.commit()
    return doc


@pytest.fixture
def workspace_repo(session: Session) -> WorkspaceRepository:
    return SQLAlchemyWorkspaceRepository(session)


@pytest.fixture
def document_repo(session: Session) -> DocumentRepository:
    return SQLAlchemyDocumentRepository(session)


@pytest.fixture
def chunk_repo(session: Session) -> ChunkRepository:
    return SQLAlchemyChunkRepository(session)


@pytest.fixture
def workspace(workspace_repo: WorkspaceRepository, session: Session) -> Workspace:
    return add_workspace(workspace_repo, session)


@pytest.fixture
def document(
    document_repo: DocumentRepository, session: Session, workspace: Workspace
) -> Document:
    return add_document(document_repo, session, workspace)


def test_workspace_add_assigns_id(
    workspace_repo: WorkspaceRepository, session: Session
):
    ws = add_workspace(workspace_repo, session, name="ws-1")
    assert ws.workspace_id is not None  # type: ignore


def test_workspace_get_returns_added(
    workspace_repo: WorkspaceRepository, session: Session
):
    ws = add_workspace(workspace_repo, session, name="ws-1")
    loaded = workspace_repo.get(ws.workspace_id)  # type: ignore
    assert loaded is not None
    assert loaded.name == "ws-1"


def test_workspace_update_changes_name(
    workspace_repo: WorkspaceRepository, session: Session
):
    ws = add_workspace(workspace_repo, session, name="old")
    ws.name = "new"
    workspace_repo.update(ws.workspace_id, ws)  # type: ignore
    session.commit()
    loaded = workspace_repo.get(ws.workspace_id)  # type: ignore
    assert loaded is not None
    assert loaded.name == "new"


def test_workspace_delete_removes(
    workspace_repo: WorkspaceRepository, session: Session
):
    ws = add_workspace(workspace_repo, session, name="x")
    workspace_repo.delete(ws.workspace_id)  # type: ignore
    session.commit()
    assert workspace_repo.get(ws.workspace_id) is None  # type: ignore


def test_document_add_assigns_id(
    document_repo: DocumentRepository, session: Session, workspace: Workspace
):
    doc = add_document(document_repo, session, workspace, title="readme")
    assert doc.document_id is not None  # type: ignore


def test_document_get_returns_with_workspace(
    document_repo: DocumentRepository, document: Document
):
    loaded = document_repo.get(document.document_id)  # type: ignore
    assert loaded is not None
    assert loaded.title == "doc"
    assert loaded.workspace.name == "ws"


def test_document_list_by_workspace(
    document_repo: DocumentRepository, session: Session, workspace: Workspace
):
    add_document(document_repo, session, workspace, title="a")
    add_document(document_repo, session, workspace, title="b")

    results, total = document_repo.list_by_workspace(
        workspace.workspace_id,  # type: ignore
        skip=0,
        limit=100,
    )

    assert len(results) == 2
    assert total == 2


def test_document_list_by_workspace_pagination(
    document_repo: DocumentRepository, session: Session, workspace: Workspace
):
    for i in range(5):
        add_document(document_repo, session, workspace, title=f"doc-{i}")

    page1, total1 = document_repo.list_by_workspace(
        workspace.workspace_id,  # type: ignore
        skip=0,
        limit=2,
    )
    page2, total2 = document_repo.list_by_workspace(
        workspace.workspace_id,  # type: ignore
        skip=2,
        limit=2,
    )
    page3, total3 = document_repo.list_by_workspace(
        workspace.workspace_id,  # type: ignore
        skip=4,
        limit=2,
    )

    assert len(page1) == 2
    assert total1 == 5
    assert len(page2) == 2
    assert total2 == 5
    assert len(page3) == 1
    assert total3 == 5
    assert page1[0].title == "doc-0"
    assert page1[1].title == "doc-1"
    assert page2[0].title == "doc-2"
    assert page2[1].title == "doc-3"
    assert page3[0].title == "doc-4"


def test_document_list_by_workspace_excludes_others(
    document_repo: DocumentRepository,
    workspace_repo: WorkspaceRepository,
    session: Session,
    workspace: Workspace,
):
    ws2 = add_workspace(workspace_repo, session, name="ws2")
    add_document(document_repo, session, ws2, title="c")

    results, total = document_repo.list_by_workspace(
        workspace.workspace_id,  # type: ignore
        skip=0,
        limit=100,
    )

    assert len(results) == 0
    assert total == 0


def test_document_delete_removes(
    document_repo: DocumentRepository, session: Session, document: Document
):
    document_repo.delete(document.document_id)  # type: ignore
    session.commit()
    assert document_repo.get(document.document_id) is None  # type: ignore


def test_delete_workspace_cascades_documents(
    document_repo: DocumentRepository,
    workspace_repo: WorkspaceRepository,
    session: Session,
    workspace: Workspace,
):
    add_document(document_repo, session, workspace, title="child")
    workspace_repo.delete(workspace.workspace_id)  # type: ignore
    session.commit()
    results, total = document_repo.list_by_workspace(
        workspace.workspace_id,  # type: ignore
        skip=0,
        limit=100,
    )
    assert results == []
    assert total == 0


def make_chunk(document: Document, content: str) -> Chunk:
    return Chunk(document_id=document.document_id, content=Content(content))  # type: ignore


def test_chunk_save_all_assigns_ids(
    chunk_repo: ChunkRepository, session: Session, document: Document
):
    chunk_repo.save_all([make_chunk(document, "hello"), make_chunk(document, "world")])
    session.commit()

    rows = list(session.execute(chunks_table.select()))

    assert len(rows) == 2


def test_chunk_save_all_persists_content(
    chunk_repo: ChunkRepository, session: Session, document: Document
):
    chunk = make_chunk(document, "hello")
    chunk_repo.save_all([chunk])
    session.commit()

    loaded = chunk_repo.get(chunk.chunk_id)  # type: ignore

    assert loaded is not None
    assert loaded.content == Content("hello")


def test_chunk_save_all_persists_embedding(
    chunk_repo: ChunkRepository, session: Session, document: Document
):
    chunk = make_chunk(document, "hello")
    chunk.add_embedding(Embedding((1.0, 2.0, 3.0)))
    chunk_repo.save_all([chunk])
    session.commit()

    loaded = chunk_repo.get(chunk.chunk_id)  # type: ignore

    assert loaded is not None
    assert loaded.has_embedding()
    assert loaded.embedding_vector == [1.0, 2.0, 3.0]


def test_chunk_duplicate_content_raises(
    chunk_repo: ChunkRepository, session: Session, document: Document
):
    chunk_repo.save_all([make_chunk(document, "dup")])
    session.commit()

    with pytest.raises(IntegrityError):
        chunk_repo.save_all([make_chunk(document, "dup")])
        session.commit()
