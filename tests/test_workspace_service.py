import pytest

from app.services.rag_service import UnknownWorkspaceError, WorkspaceService
from tests.conftest import FakeUnitOfWork


@pytest.fixture
def uow() -> FakeUnitOfWork:
    return FakeUnitOfWork()


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
    workspace = workspace_service.get_workspace(999)
    assert workspace is None


def test_get_workspaces_paginates_and_returns_total(
    workspace_service: WorkspaceService,
) -> None:
    workspace_service.new_workspace("WS 1")
    workspace_service.new_workspace("WS 2")
    workspace_service.new_workspace("WS 3")

    workspaces, total = workspace_service.get_workspaces(skip=1, limit=1)
    assert total == 3
    assert len(workspaces) == 1
    assert workspaces[0].name == "WS 2"


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
    with pytest.raises(UnknownWorkspaceError):
        workspace_service.new_document(999, "Title", "Some text content")
