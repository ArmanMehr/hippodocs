from io import BytesIO
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from httpx import Response

from app.api.dependencies import (
    get_file_readers,
    get_ingestion_service,
    get_rag_service,
    get_workspace_service,
)
from app.main import app
from app.services.factory import (
    create_file_readers,
    create_ingestion_service,
    create_rag_service,
    create_workspace_service,
)
from tests.conftest import (
    FakeEmbedder,
    FakeLLMChat,
    FakeUnitOfWork,
)


@pytest.fixture
def client() -> TestClient:
    def _mock_get(*args: Any, **kwargs: Any) -> Response:
        return Response(status_code=200)

    httpx.get = _mock_get
    uow = FakeUnitOfWork()

    workspace_service = create_workspace_service(uow=uow)
    ingestion_service = create_ingestion_service(uow=uow)
    rag_service = create_rag_service(
        uow=uow, embedder=FakeEmbedder(), llm=FakeLLMChat()
    )
    file_readers = create_file_readers()

    app.dependency_overrides[get_workspace_service] = lambda: workspace_service
    app.dependency_overrides[get_ingestion_service] = lambda: ingestion_service
    app.dependency_overrides[get_rag_service] = lambda: rag_service
    app.dependency_overrides[get_file_readers] = lambda: file_readers

    return TestClient(app)


def test_create_workspace(client: TestClient) -> None:
    response = client.post("/v1/workspaces/", json={"name": "Test Workspace"})
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Test Workspace"
    assert "workspace_id" in data


def test_list_workspaces(client: TestClient) -> None:
    client.post("/v1/workspaces/", json={"name": "WS 1"})
    client.post("/v1/workspaces/", json={"name": "WS 2"})

    response = client.get("/v1/workspaces/")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert len(data["workspaces"]) == 2


def test_get_workspace(client: TestClient) -> None:
    create_resp = client.post("/v1/workspaces/", json={"name": "My Workspace"})
    ws_id = create_resp.json()["workspace_id"]

    response = client.get(f"/v1/workspaces/{ws_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "My Workspace"
    assert data["workspace_id"] == ws_id


def test_get_workspace_not_found(client: TestClient) -> None:
    response = client.get("/v1/workspaces/999")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_delete_workspace(client: TestClient) -> None:
    create_resp = client.post("/v1/workspaces/", json={"name": "To Delete"})
    ws_id = create_resp.json()["workspace_id"]

    response = client.delete(f"/v1/workspaces/{ws_id}")
    assert response.status_code == 204

    get_resp = client.get(f"/v1/workspaces/{ws_id}")
    assert get_resp.status_code == 404


def test_upload_document(client: TestClient) -> None:
    create_resp = client.post("/v1/workspaces/", json={"name": "WS with Docs"})
    ws_id = create_resp.json()["workspace_id"]

    file_content = b"Hello world. This is a test document."
    files = {"file": ("test.md", BytesIO(file_content), "text/markdown")}
    response = client.post(f"/v1/workspaces/{ws_id}/documents", files=files)

    assert response.status_code == 201
    data = response.json()
    assert data["title"] == "test"
    assert "document_id" in data


def test_list_documents_in_workspace(client: TestClient) -> None:
    create_resp = client.post("/v1/workspaces/", json={"name": "WS for List"})
    ws_id = create_resp.json()["workspace_id"]

    for i in range(3):
        files = {
            "file": (f"doc{i}.md", BytesIO(f"Content {i}".encode()), "text/markdown")
        }
        client.post(f"/v1/workspaces/{ws_id}/documents", files=files)

    response = client.get(f"/v1/workspaces/{ws_id}/documents")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 3
    assert len(data["documents"]) == 3


def test_delete_document(client: TestClient) -> None:
    create_resp = client.post("/v1/workspaces/", json={"name": "WS for Delete"})
    ws_id = create_resp.json()["workspace_id"]

    files = {"file": ("test.md", BytesIO(b"Content to delete"), "text/markdown")}
    upload_resp = client.post(f"/v1/workspaces/{ws_id}/documents", files=files)
    doc_id = upload_resp.json()["document_id"]

    response = client.delete(f"/v1/workspaces/{ws_id}/documents/{doc_id}")
    assert response.status_code == 204

    list_resp = client.get(f"/v1/workspaces/{ws_id}/documents")
    assert list_resp.json()["total"] == 0


def test_ask_question(client: TestClient) -> None:
    create_resp = client.post("/v1/workspaces/", json={"name": "WS for Ask"})
    ws_id = create_resp.json()["workspace_id"]

    files = {
        "file": ("test.md", BytesIO(b"HippoDocs is a RAG platform."), "text/markdown")
    }
    client.post(f"/v1/workspaces/{ws_id}/documents", files=files)

    response = client.post(
        f"/v1/workspaces/{ws_id}/ask", json={"question": "What is HippoDocs?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "content" in data
