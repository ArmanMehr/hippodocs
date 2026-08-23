import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.adapters.orm import chunks_table
from app.adapters.repository import (
    ChunkRepository,
    DocumentRepository,
    WorkspaceRepository,
)
from app.domain.models import Chunk, Content, Document, Embedding, Workspace


@pytest.fixture
def workspace_repo(session: Session):
    from app.adapters.repository import SQLAlchemyWorkspaceRepository

    return SQLAlchemyWorkspaceRepository(session)


@pytest.fixture
def document_repo(session: Session):
    from app.adapters.repository import SQLAlchemyDocumentRepository

    return SQLAlchemyDocumentRepository(session)


@pytest.fixture
def chunk_repo(session: Session):
    from app.adapters.repository import SQLAlchemyChunkRepository

    return SQLAlchemyChunkRepository(session)


@pytest.fixture
def workspace(session: Session, workspace_repo: WorkspaceRepository) -> Workspace:
    ws = Workspace(name="ws")
    workspace_repo.add(ws)
    session.commit()
    return ws


@pytest.fixture
def document(
    session: Session, document_repo: DocumentRepository, workspace: Workspace
) -> Document:
    doc = Document(content=Content("text"), workspace=workspace, title="doc")
    document_repo.add(doc)
    session.commit()
    return doc


def test_workspace_add_assigns_id(
    workspace_repo: WorkspaceRepository, session: Session
):
    ws = Workspace(name="ws-1")
    workspace_repo.add(ws)
    session.commit()
    assert ws.workspace_id is not None  # type: ignore


def test_workspace_get_returns_added(
    workspace_repo: WorkspaceRepository, session: Session
):
    ws = Workspace(name="ws-1")
    workspace_repo.add(ws)
    session.commit()
    loaded = workspace_repo.get(ws.workspace_id)  # type: ignore
    assert loaded.name == "ws-1"


def test_workspace_update_changes_name(
    workspace_repo: WorkspaceRepository, session: Session
):
    ws = Workspace(name="old")
    workspace_repo.add(ws)
    session.commit()
    ws.name = "new"
    workspace_repo.update(ws.workspace_id, ws)  # type: ignore
    session.commit()
    assert workspace_repo.get(ws.workspace_id).name == "new"  # type: ignore


def test_workspace_delete_removes(
    workspace_repo: WorkspaceRepository, session: Session
):
    ws = Workspace(name="x")
    workspace_repo.add(ws)
    session.commit()
    workspace_repo.delete(ws.workspace_id)  # type: ignore
    session.commit()
    assert workspace_repo.get(ws.workspace_id) is None  # type: ignore


def test_document_add_assigns_id(
    document_repo: DocumentRepository, session: Session, workspace: Workspace
):
    doc = Document(content=Content("text"), workspace=workspace, title="readme")
    document_repo.add(doc)
    session.commit()
    assert doc.document_id is not None  # type: ignore


def test_document_get_returns_with_workspace(
    document_repo: DocumentRepository, document: Document
):
    loaded = document_repo.get(document.document_id)  # type: ignore
    assert loaded.title == "doc"
    assert loaded.workspace.name == "ws"


def test_document_list_by_workspace(
    document_repo: DocumentRepository,
    session: Session,
    workspace: Workspace,
):
    document_repo.add(Document(content=Content("text"), workspace=workspace, title="a"))
    document_repo.add(Document(content=Content("text"), workspace=workspace, title="b"))
    session.commit()
    results = document_repo.list_by_workspace(workspace.workspace_id)  # type: ignore
    assert len(results) == 2


def test_document_list_by_workspace_excludes_others(
    document_repo: DocumentRepository,
    session: Session,
    workspace: Workspace,
    workspace_repo: WorkspaceRepository,
):
    ws2 = Workspace(name="ws2")
    workspace_repo.add(ws2)
    session.commit()
    document_repo.add(Document(content=Content("text"), workspace=ws2, title="c"))
    session.commit()
    results = document_repo.list_by_workspace(workspace.workspace_id)  # type: ignore
    assert len(results) == 0


def test_document_delete_removes(
    document_repo: DocumentRepository,
    session: Session,
    document: Document,
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
    document_repo.add(
        Document(content=Content("text"), workspace=workspace, title="child")
    )
    session.commit()
    workspace_repo.delete(workspace.workspace_id)  # type: ignore
    session.commit()
    assert document_repo.list_by_workspace(workspace.workspace_id) == []  # type: ignore


def test_chunk_save_all_assigns_ids(
    chunk_repo: ChunkRepository,
    session: Session,
    document: Document,
):
    c1 = Chunk(document_id=document.document_id, content=Content("hello"))  # type: ignore
    c2 = Chunk(document_id=document.document_id, content=Content("world"))  # type: ignore
    chunk_repo.save_all([c1, c2])
    session.commit()
    rows = list(session.execute(chunks_table.select()))
    assert len(rows) == 2


def test_chunk_save_all_persists_content(
    chunk_repo: ChunkRepository,
    session: Session,
    document: Document,
):
    chunk = Chunk(document_id=document.document_id, content=Content("hello"))  # type: ignore
    chunk_repo.save_all([chunk])
    session.commit()
    loaded = chunk_repo.get(chunk.chunk_id)  # type: ignore
    assert loaded.content == Content("hello")


def test_chunk_save_all_persists_embedding(
    chunk_repo: ChunkRepository,
    session: Session,
    document: Document,
):
    chunk = Chunk(document_id=document.document_id, content=Content("hello"))  # type: ignore
    chunk.add_embedding(Embedding((1.0, 2.0, 3.0)))
    chunk_repo.save_all([chunk])
    session.commit()
    loaded = chunk_repo.get(chunk.chunk_id)  # type: ignore
    assert loaded.has_embedding()
    assert loaded.embedding_vector == [1.0, 2.0, 3.0]


def test_chunk_duplicate_content_raises(
    chunk_repo: ChunkRepository,
    session: Session,
    document: Document,
):
    chunk_repo.save_all(
        [Chunk(document_id=document.document_id, content=Content("dup"))]  # type: ignore
    )
    session.commit()
    with pytest.raises(IntegrityError):
        chunk_repo.save_all(
            [Chunk(document_id=document.document_id, content=Content("dup"))]  # type: ignore
        )
        session.commit()
