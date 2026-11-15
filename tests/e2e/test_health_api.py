from fastapi import status
from fastapi.testclient import TestClient
from pytest_mock import MockerFixture

from app.health import ProviderHealth
from app.main import app


def _mock_providers_health(
    mocker: MockerFixture, *, embedding: bool, llm: bool
) -> None:
    mocker.patch(
        "app.api.v1.health.check_providers_health",
        new_callable=mocker.AsyncMock,
        return_value=ProviderHealth(embedding=embedding, llm=llm),
    )


def test_health_live():
    client = TestClient(app)
    response = client.get("/v1/health/live")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "ok"}


def test_health_ready_all_healthy(mocker: MockerFixture):
    client = TestClient(app)
    _mock_providers_health(mocker, embedding=True, llm=True)
    mocker.patch(
        "app.api.v1.health.check_db_health",
        return_value=True,
    )
    response = client.get("/v1/health/ready")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "ready"
    assert data["database"] is True
    assert data["providers"] == {"embedding": True, "llm": True}


def test_health_ready_provider_unhealthy(mocker: MockerFixture):
    client = TestClient(app)
    _mock_providers_health(mocker, embedding=True, llm=False)
    mocker.patch(
        "app.api.v1.health.check_db_health",
        return_value=True,
    )
    response = client.get("/v1/health/ready")
    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    data = response.json()
    assert data["status"] == "unhealthy"
    assert data["database"] is True
    assert data["providers"] == {"embedding": True, "llm": False}


def test_health_ready_db_unhealthy(mocker: MockerFixture):
    client = TestClient(app)
    _mock_providers_health(mocker, embedding=True, llm=True)
    mocker.patch(
        "app.api.v1.health.check_db_health",
        return_value=False,
    )
    response = client.get("/v1/health/ready")
    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    data = response.json()
    assert data["status"] == "unhealthy"
    assert data["database"] is False
    assert data["providers"] == {"embedding": True, "llm": True}
