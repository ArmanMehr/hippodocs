from collections.abc import Callable
from typing import Any

import httpx
import pytest
from pytest_mock import MockerFixture

from app.configs import Configs
from app.health import (
    check_ollama_health,
    check_openai_health,
    check_providers_health,
)


def _patch_async_client(
    mocker: MockerFixture,
    handler: Callable[[httpx.Request], httpx.Response],
) -> None:
    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    mocker.patch("httpx.AsyncClient", side_effect=factory)


@pytest.mark.anyio
async def test_check_ollama_health_success():
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://localhost:11434/"
        return httpx.Response(200, text="Ollama is running")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        is_healthy = await check_ollama_health("http://localhost:11434", client=client)
    assert is_healthy is True


@pytest.mark.anyio
async def test_check_ollama_health_network_error():
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        is_healthy = await check_ollama_health("http://localhost:11434", client=client)
    assert is_healthy is False


@pytest.mark.anyio
async def test_check_openai_health_success(mocker: MockerFixture):
    mocker.patch("app.health._settings", Configs(OPENAI_API_KEY="sk-test-key"))

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://api.openai.com/v1/models"
        assert request.headers.get("Authorization") == "Bearer sk-test-key"
        return httpx.Response(200, json={"data": []})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        is_healthy = await check_openai_health(
            "https://api.openai.com/v1", client=client
        )
    assert is_healthy is True


@pytest.mark.anyio
async def test_check_openai_health_unauthorized():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "Invalid API key"}})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        is_healthy = await check_openai_health(
            "https://api.openai.com/v1", client=client
        )
    assert is_healthy is False


@pytest.mark.anyio
async def test_check_providers_health_all_healthy(
    mocker: MockerFixture,
) -> None:
    settings = Configs(
        EMBEDDING_PROVIDER="openai",
        EMBEDDING_BASE_URL="http://localhost:8000",
        LLM_PROVIDER="openai",
        LLM_BASE_URL="https://api.openai.com/v1",
        OPENAI_API_KEY="sk-test",
    )
    mocker.patch("app.health._settings", settings)

    def handler(request: httpx.Request) -> httpx.Response:
        if "localhost" in request.url.host:
            return httpx.Response(200, text="Ollama is running")
        if "api.openai.com" in request.url.host:
            return httpx.Response(200, json={"data": []})
        return httpx.Response(404)

    _patch_async_client(mocker, handler)

    health = await check_providers_health()

    assert health.get_dict() == {"embedding": True, "llm": True}
    assert health.is_healthy() is True


@pytest.mark.anyio
async def test_check_providers_health_one_unhealthy(
    mocker: MockerFixture,
) -> None:
    settings = Configs(
        EMBEDDING_PROVIDER="openai",
        EMBEDDING_BASE_URL="http://localhost:8000",
        LLM_PROVIDER="openai",
        LLM_BASE_URL="https://api.openai.com/v1",
        OPENAI_API_KEY="sk-test",
    )
    mocker.patch("app.health._settings", settings)

    def handler(request: httpx.Request) -> httpx.Response:
        if "localhost" in request.url.host:
            return httpx.Response(200, text="Ollama is running")
        if "api.openai.com" in request.url.host:
            return httpx.Response(401, json={"error": "unauthorized"})
        return httpx.Response(404)

    _patch_async_client(mocker, handler)

    health = await check_providers_health()

    assert health.get_dict() == {"embedding": True, "llm": False}
    assert health.is_healthy() is False
