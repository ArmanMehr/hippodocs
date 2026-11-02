import os

from fastapi.testclient import TestClient
from pytest_mock import MockerFixture
from sqlalchemy.exc import OperationalError

from app.api.dependencies import get_workspace_service
from app.main import app

os.environ["LANGFUSE_TRACING_ENABLED"] = "false"
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""
os.environ["LANGFUSE_HOST"] = ""
os.environ["LANGFUSE_BASE_URL"] = ""


def test_happy_path_single_info_log_with_request_id(
    client: TestClient, mocker: MockerFixture
) -> None:
    logger = mocker.patch("app.main.logger.info")
    response = client.get("/v1/workspaces/")

    assert response.status_code == 200
    assert response.headers.get("X-Request-Id") is not None
    logger.assert_called_once()


def test_validation_error_returns_422_and_error_payload(
    client: TestClient, mocker: MockerFixture
):
    info_logger = mocker.patch("app.main.logger.info")
    response = client.post("/v1/workspaces/", json={})

    assert response.status_code == 422
    assert response.headers.get("X-Request-Id") is not None

    data = response.json()

    assert "detail" in data
    assert data["error_code"] == "request_validation_error"
    info_logger.assert_called_once()


def test_missing_workspace_id_returns_404_and_error_payload(
    client: TestClient, mocker: MockerFixture
):
    info_logger = mocker.patch("app.main.logger.info")
    response = client.get("/v1/workspaces/999")

    assert response.status_code == 404
    assert response.headers.get("X-Request-Id") is not None

    data = response.json()

    assert "detail" in data
    assert data["error_code"] == "workspace_not_found"
    info_logger.assert_called_once()


def test_database_error_returns_503_and_sanitized_payload(
    client: TestClient, mocker: MockerFixture
) -> None:
    error_logger = mocker.patch("app.main.logger.error")

    class BrokenWorkspaceService:
        def get_workspace(self, _: int):
            raise OperationalError(
                "SELECT 1", params={}, orig=Exception("Connection refused")
            )

    app.dependency_overrides[get_workspace_service] = lambda: BrokenWorkspaceService()
    response = client.get("/v1/workspaces/1")
    app.dependency_overrides.clear()

    assert response.status_code == 503

    data = response.json()

    assert data["error_code"] == "database_unavailable"
    assert "SELECT 1" not in response.text
    error_logger.assert_called()
    assert error_logger.call_args.kwargs["reason"] == "Connection refused"


def test_unexpected_runtime_error_returns_500_and_sanitized_payload(
    client: TestClient, mocker: MockerFixture
) -> None:
    exception_logger = mocker.patch("app.main.logger.exception")

    class BuggyWorkspaceService:
        def get_workspace(self, workspace_id: int):
            raise RuntimeError("Unexpected internal crash or bug!")

    app.dependency_overrides[get_workspace_service] = lambda: BuggyWorkspaceService()
    response = client.get("/v1/workspaces/1")
    app.dependency_overrides.clear()

    assert response.status_code == 500
    assert response.headers.get("X-Request-ID") is not None

    data = response.json()

    assert data["error_code"] == "internal_error"
    assert data["detail"] == "Internal server error"
    assert "Unexpected internal crash" not in response.text
    exception_logger.assert_called_once()
