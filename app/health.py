import asyncio

import httpx
import structlog
from sqlalchemy import text

from app.configs import Configs, get_settings
from app.exceptions import UnsupportedProvider
from app.services.uow import SQLAlchemyUnitOfWork

logger = structlog.get_logger(__name__)


def check_db_health() -> bool:
    try:
        uow = SQLAlchemyUnitOfWork()
        with uow:
            uow.session.execute(text("SELECT 1"))
        return True
    except Exception:  # noqa
        return False


async def check_ollama_health(
    base_url: str, client: httpx.AsyncClient | None = None
) -> bool:
    target_url = f"{base_url.rstrip('/')}/"
    try:
        if client is not None:
            resp = await client.get(target_url)
        else:
            async with httpx.AsyncClient(timeout=3.0) as local_client:
                resp = await local_client.get(target_url)
        return resp.status_code == 200
    except httpx.RequestError:
        return False


async def check_openai_health(
    base_url: str, api_key: str, client: httpx.AsyncClient | None = None
) -> bool:
    target_url = f"{base_url.rstrip('/')}/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        if client is not None:
            resp = await client.get(target_url, headers=headers)
        else:
            async with httpx.AsyncClient(timeout=3.0) as local_client:
                resp = await local_client.get(target_url, headers=headers)
        return resp.status_code == 200
    except httpx.RequestError:
        return False


class CheckProvidersHealth:
    def __init__(self, configs: Configs | None = None) -> None:
        self.configs = configs or get_settings()

    async def _check_embedder_health(
        self, client: httpx.AsyncClient | None = None
    ) -> bool:
        match self.configs.EMBEDDING_PROVIDER:
            case "openai":
                return await check_openai_health(
                    self.configs.EMBEDDING_BASE_URL,
                    self.configs.OPENAI_API_KEY,
                    client=client,
                )
            case _:
                return False

    async def _check_llm_health(self, client: httpx.AsyncClient | None = None) -> bool:
        match self.configs.LLM_PROVIDER:
            case "ollama":
                return await check_ollama_health(
                    self.configs.LLM_BASE_URL, client=client
                )
            case "openai":
                return await check_openai_health(
                    self.configs.LLM_BASE_URL,
                    self.configs.OPENAI_API_KEY,
                    client=client,
                )
            case _:
                raise UnsupportedProvider()

    async def check(self) -> dict[str, bool]:
        async with httpx.AsyncClient(timeout=3.0) as client:
            embedder_healthy, llm_healthy = await asyncio.gather(
                self._check_embedder_health(client=client),
                self._check_llm_health(client=client),
            )
        return {
            "embedder": embedder_healthy,
            "llm": llm_healthy,
        }


async def check_providers_health():
    provider_health = CheckProvidersHealth()
    health_details = await provider_health.check()
    if not all(health_details.values()):
        logger.warning(
            "provider_health_check_failed",
            details=health_details,
        )
    else:
        logger.info("provider_health_check_passed", details=health_details)
