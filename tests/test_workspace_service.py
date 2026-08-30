import pytest

from app.services.rag_service import WorkspaceNotFound, WorkspaceService
from tests.conftest import FakeUnitOfWork


@pytest.fixture
def workspace_service(uow: FakeUnitOfWork) -> WorkspaceService:
    return WorkspaceService(uow=uow)


def test_new_workspace_returns_assigned_id(
    workspace_service: WorkspaceService,
) -> None:
    workspace_id = workspace_service.new_workspace("Test Workspace")
    assert workspace_id == 1


def test_get_workspace_returns_workspace(workspace_service: WorkspaceService) -> None:
    workspace_id = workspace_service.new_workspace("Test Workspace")
    workspace = workspace_service.get_workspace(workspace_id)
    assert workspace is not None
    assert workspace.name == "Test Workspace"


def test_get_workspace_returns_none_for_unknown_id(
    workspace_service: WorkspaceService,
) -> None:
    assert workspace_service.get_workspace(999) is None


def test_get_workspaces_paginates_and_returns_total(
    workspace_service: WorkspaceService,
) -> None:
    for name in ("WS 1", "WS 2", "WS 3"):
        workspace_service.new_workspace(name)

    workspaces, total = workspace_service.get_workspaces(skip=1, limit=1)
    assert total == 3
    assert [ws.name for ws in workspaces] == ["WS 2"]


def test_delete_workspace_removes_workspace(
    workspace_service: WorkspaceService,
) -> None:
    workspace_id = workspace_service.new_workspace("To Delete")
    workspace_service.delete_workspace(workspace_id)
    assert workspace_service.get_workspace(workspace_id) is None


def test_new_document_creates_doc_and_returns_id(
    workspace_service: WorkspaceService,
) -> None:
    workspace_id = workspace_service.new_workspace("WS")
    doc_id = workspace_service.new_document(workspace_id, "Title", "Some text content")
    assert doc_id == 1


def test_new_document_for_unknown_workspace(
    workspace_service: WorkspaceService,
) -> None:
    with pytest.raises(WorkspaceNotFound):
        workspace_service.new_document(999, "Title", "Some text content")


def test_list_documents_in_workspace_paginates_and_returns_total(
    workspace_service: WorkspaceService,
) -> None:
    workspace_id = workspace_service.new_workspace("WS")
    workspace_service.new_document(workspace_id, "Doc 1", "text 1")
    workspace_service.new_document(workspace_id, "Doc 2", "text 2")
    workspace_service.new_document(workspace_id, "Doc 3", "text 3")

    documents, total = workspace_service.list_documents_in_workspace(
        workspace_id, skip=1, limit=1
    )
    assert total == 3
    assert [doc.title for doc in documents] == ["Doc 2"]
