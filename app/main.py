import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from sqlalchemy.exc import InterfaceError, OperationalError
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from app import REQUEST_ID_REGEX, configure_logging
from app.adapters.orm import start_mappers
from app.api.v1 import router as api_v1_router
from app.configs import get_settings
from app.exceptions import AppError, DatabaseUnavailable, error_payload
from app.limiter import limiter
from app.observability import flush_langfuse, initialize_langfuse

level = getattr(logging, get_settings().LOG_LEVEL)
json_logs = get_settings().ENV == "prd"
configure_logging(level=level, json_logs=json_logs)

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("api_starting")
    start_mappers()
    initialize_langfuse()
    try:
        yield
    finally:
        logger.info("api_stopping")
        flush_langfuse()


app = FastAPI(lifespan=lifespan, title="Simple RAG API", version="1.0.0")
app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore


@app.middleware("http")
async def request_logging_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    structlog.contextvars.clear_contextvars()
    request_id = request.headers.get("X-Request-ID", "").strip()
    if not request_id or not REQUEST_ID_REGEX.fullmatch(request_id):
        request_id = uuid.uuid4().hex

    structlog.contextvars.bind_contextvars(
        request_id=request_id,
    )
    start_time = time.perf_counter()
    try:
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info(
            "http_request_completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
        )
        response.headers["X-Request-ID"] = request_id
        return response

    except (OperationalError, InterfaceError) as exc:
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.warning(
            "database_unavailable",
            method=request.method,
            path=request.url.path,
            duration_ms=duration_ms,
            reason=str(exc.orig),
        )
        error = DatabaseUnavailable()
        response = JSONResponse(
            status_code=error.status_code, content=error_payload(error)
        )
        response.headers["X-Request-ID"] = request_id
        return response

    except Exception:
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.exception(
            "http_request_failed",
            method=request.method,
            path=request.url.path,
            duration_ms=duration_ms,
        )
        response = JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error",
                "error_code": "internal_error",
            },
        )
        response.headers["X-Request-ID"] = request_id
        return response

    finally:
        structlog.contextvars.clear_contextvars()


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exception: AppError) -> JSONResponse:
    logger.warning(
        "application_error",
        error_code=exception.error_code,
        method=request.method,
        path=request.url.path,
        status_code=exception.status_code,
    )
    return JSONResponse(
        status_code=exception.status_code, content=error_payload(exception)
    )


app.include_router(api_v1_router)


@app.get("/health", tags=["health"])
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    if get_settings().ENV == "dev":
        import uvicorn

        uvicorn.run(
            "app.main:app", host="0.0.0.0", port=8000, reload=False, access_log=False
        )
