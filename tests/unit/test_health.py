from collections.abc import Callable
from typing import Any

import httpx
import pytest
from pytest_mock import MockerFixture

from app.configs import Configs
from app.health import check_providers_health


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
