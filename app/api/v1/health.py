import structlog
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from app.health import CheckProvidersHealth, check_db_health
from app.schemas import ReadinessResponse

logger = structlog.get_logger()

router = APIRouter(tags=["health"])


@router.get("/healthz", status_code=status.HTTP_200_OK)
def liveness() -> JSONResponse:
    return JSONResponse(content={"status": "ok"})


@router.get("/readyz", response_model=ReadinessResponse)
async def readiness() -> JSONResponse:
    provider_health = CheckProvidersHealth()
    provider_details = await provider_health.check()
    db_ok = check_db_health()

    all_providers_ok = all(provider_details.values())
    is_ready = db_ok and all_providers_ok

    payload = ReadinessResponse(
        status="ready" if is_ready else "unhealthy",
        database=db_ok,
        providers=provider_details,
    ).model_dump()

    status_code = (
        status.HTTP_200_OK if is_ready else status.HTTP_503_SERVICE_UNAVAILABLE
    )
    return JSONResponse(status_code=status_code, content=payload)
