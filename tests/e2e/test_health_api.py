from fastapi import status
from fastapi.testclient import TestClient
from pytest_mock import MockerFixture

from app.main import app


def test_healthz_liveness():
    client = TestClient(app)
    response = client.get("/v1/healthz")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "ok"}


def test_readyz_all_healthy(mocker: MockerFixture):
    client = TestClient(app)
    mocker.patch(
        "app.api.v1.health.CheckProvidersHealth.check",
        new_callable=mocker.AsyncMock,
        return_value={"embedder": True, "llm": True},
    )
    mocker.patch(
        "app.api.v1.health.check_db_health",
        return_value=True,
    )
    response = client.get("/v1/readyz")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "ready"
    assert data["database"] is True
    assert data["providers"] == {"embedder": True, "llm": True}


def test_readyz_provider_unhealthy(mocker: MockerFixture):
    client = TestClient(app)
    mocker.patch(
        "app.api.v1.health.CheckProvidersHealth.check",
        new_callable=mocker.AsyncMock,
        return_value={"embedder": True, "llm": False},
    )
    mocker.patch(
        "app.api.v1.health.check_db_health",
        return_value=True,
    )
    response = client.get("/v1/readyz")
    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    data = response.json()
    assert data["status"] == "unhealthy"
    assert data["database"] is True
    assert data["providers"] == {"embedder": True, "llm": False}


def test_readyz_db_unhealthy(mocker: MockerFixture):
    client = TestClient(app)
    mocker.patch(
        "app.api.v1.health.CheckProvidersHealth.check",
        new_callable=mocker.AsyncMock,
        return_value={"embedder": True, "llm": True},
    )
    mocker.patch(
        "app.api.v1.health.check_db_health",
        return_value=False,
    )
    response = client.get("/v1/readyz")
    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    data = response.json()
    assert data["status"] == "unhealthy"
    assert data["database"] is False
