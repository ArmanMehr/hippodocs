import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass

import httpx
from sqlalchemy import text

from app.configs import get_settings
from app.services.uow import SQLAlchemyUnitOfWork

_settings = get_settings()


@dataclass(frozen=True)
class ProviderHealth:
    embedding: bool
    llm: bool

    def get_dict(self) -> dict[str, bool]:
        return asdict(self)

    def is_healthy(self) -> bool:
        return all((self.embedding, self.llm))


async def check_ollama_health(
    base_url: str,
    client: httpx.AsyncClient,
) -> bool:
    target_url = f"{base_url.rstrip('/')}/"
    try:
        resp = await client.get(target_url)
        return resp.status_code == 200
    except httpx.RequestError:
        return False


async def check_openai_health(
    base_url: str,
    client: httpx.AsyncClient,
) -> bool:
    target_url = f"{base_url.rstrip('/')}/models"
    headers = {"Authorization": f"Bearer {_settings.OPENAI_API_KEY}"}
    try:
        resp = await client.get(target_url, headers=headers)
        return resp.status_code == 200
    except httpx.RequestError:
        return False


_PROVIDER_HEALTH_CHECKER: dict[str, Callable[..., Awaitable[bool]]] = {
    "ollama": check_ollama_health,
    "openai": check_openai_health,
}


async def check_providers_health() -> ProviderHealth:
    async with httpx.AsyncClient(timeout=3.0) as client:
        embedder_healthy, llm_healthy = await asyncio.gather(
            _PROVIDER_HEALTH_CHECKER[_settings.EMBEDDING_PROVIDER](
                _settings.EMBEDDING_BASE_URL, client
            ),
            _PROVIDER_HEALTH_CHECKER[_settings.LLM_PROVIDER](
                _settings.LLM_BASE_URL, client
            ),
        )
    return ProviderHealth(embedder_healthy, llm_healthy)


def check_db_health() -> bool:
    try:
        uow = SQLAlchemyUnitOfWork()
        with uow:
            uow.session.execute(text("SELECT 1"))
        return True
    except Exception:  # noqa
        return False
