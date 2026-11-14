from typing import Any

import httpx
import pytest

from app.configs import Configs
from app.health import (
    CheckProvidersHealth,
    check_ollama_health,
    check_openai_health,
)


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
async def test_check_ollama_health_failure_status_code():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Server Error")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        is_healthy = await check_ollama_health("http://localhost:11434", client=client)
    assert is_healthy is False


@pytest.mark.anyio
async def test_check_ollama_health_network_error():
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        is_healthy = await check_ollama_health("http://localhost:11434", client=client)
    assert is_healthy is False


@pytest.mark.anyio
async def test_check_openai_health_success():
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://api.openai.com/v1/models"
        assert request.headers.get("Authorization") == "Bearer sk-test-key"
        return httpx.Response(200, json={"data": []})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        is_healthy = await check_openai_health(
            "https://api.openai.com/v1", "sk-test-key", client=client
        )
    assert is_healthy is True


@pytest.mark.anyio
async def test_check_openai_health_unauthorized():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "Invalid API key"}})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        is_healthy = await check_openai_health(
            "https://api.openai.com/v1", "wrong-key", client=client
        )
    assert is_healthy is False


@pytest.mark.anyio
async def test_check_openai_health_timeout():
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("Read timed out")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        is_healthy = await check_openai_health(
            "https://api.openai.com/v1", "sk-test-key", client=client
        )
    assert is_healthy is False


@pytest.mark.anyio
async def test_check_provider_health_all_healthy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Configs(
        EMBEDDING_PROVIDER="openai",
        EMBEDDING_BASE_URL="http://localhost:8000",
        LLM_PROVIDER="openai",
        LLM_BASE_URL="https://api.openai.com/v1",
        OPENAI_API_KEY="sk-test",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if "localhost" in request.url.host:
            return httpx.Response(200, text="Ollama is running")
        elif "api.openai.com" in request.url.host:
            return httpx.Response(200, json={"data": []})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def mocked_async_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", mocked_async_client)

    checker = CheckProvidersHealth(configs=settings)
    details = await checker.check()

    assert details == {"embedder": True, "llm": True}
    assert all((await checker.check()).values()) is True


@pytest.mark.anyio
async def test_check_provider_health_one_unhealthy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Configs(
        EMBEDDING_PROVIDER="openai",
        EMBEDDING_BASE_URL="http://localhost:8000",
        LLM_PROVIDER="openai",
        LLM_BASE_URL="https://api.openai.com/v1",
        OPENAI_API_KEY="sk-test",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if "localhost" in request.url.host:
            return httpx.Response(200, text="Ollama is running")
        elif "api.openai.com" in request.url.host:
            return httpx.Response(401, json={"error": "unauthorized"})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def mocked_async_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", mocked_async_client)

    checker = CheckProvidersHealth(configs=settings)
    details = await checker.check()

    assert details == {"embedder": True, "llm": False}
    assert all((await checker.check()).values()) is False
