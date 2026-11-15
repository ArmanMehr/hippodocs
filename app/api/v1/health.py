from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from app.health import check_db_health, check_providers_health
from app.schemas import ReadinessResponse

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live", status_code=status.HTTP_200_OK)
async def liveness():
    return JSONResponse(content={"status": "ok"})


@router.get("/ready", response_model=ReadinessResponse)
async def readiness() -> JSONResponse:
    providers_health = (await check_providers_health()).get_dict()
    db_ok = check_db_health()

    is_ready = all(providers_health.values()) and db_ok
    payload = ReadinessResponse(
        status="ready" if is_ready else "unhealthy",
        database=db_ok,
        providers=providers_health,
    ).model_dump()

    status_code = (
        status.HTTP_200_OK if is_ready else status.HTTP_503_SERVICE_UNAVAILABLE
    )
    return JSONResponse(status_code=status_code, content=payload)
