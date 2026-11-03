import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from langfuse import propagate_attributes
from slowapi import _rate_limit_exceeded_handler
from sqlalchemy.exc import InterfaceError, OperationalError
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from app import REQUEST_ID_REGEX, configure_logging
from app.adapters.orm import start_mappers
from app.api.v1 import router as api_v1_router
from app.configs import get_settings
from app.exceptions import (
    ClientError,
    DatabaseUnavailable,
    RateLimitError,
    ServerError,
    error_payload,
    get_fastapi_exception_payload,
)
from app.limiter import limiter
from app.observability import flush_langfuse, initialize_langfuse

level = getattr(logging, get_settings().LOG_LEVEL)
configure_logging(level=level, json_logs=get_settings().LOG_JSON)

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
app.include_router(api_v1_router)

# FIX: Change the trusted host for deployment
app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")
app.state.limiter = limiter
app.add_exception_handler(RateLimitError, _rate_limit_exceeded_handler)  # type: ignore


@app.middleware("http")
async def request_logging_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    structlog.contextvars.clear_contextvars()
    request_id = request.headers.get("X-Request-ID", "").strip()
    if not REQUEST_ID_REGEX.fullmatch(request_id):
        request_id = uuid.uuid4().hex

    structlog.contextvars.bind_contextvars(request_id=request_id)
    try:
        # TODO: Apply session_id for multi-turn chat sessions
        propagate_attributes(metadata={"request_id": request_id})
    except Exception:  # noqa
        pass

    start_time = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        structlog.contextvars.bind_contextvars(error_code="internal_error")
        logger.exception("unhandled_server_error")
        error = ServerError(detail="Internal server error")
        response = JSONResponse(
            status_code=error.status_code, content=error_payload(error)
        )

    logger.info(
        "http_request_completed",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=round((time.perf_counter() - start_time) * 1000, 2),
    )
    response.headers["X-Request-ID"] = request_id
    structlog.contextvars.clear_contextvars()
    return response


@app.exception_handler(OperationalError)
@app.exception_handler(InterfaceError)
async def database_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    error_code = "database_unavailable"
    structlog.contextvars.bind_contextvars(error_code=error_code)
    logger.error(error_code, reason=str(getattr(exc, "orig", exc)))
    error = DatabaseUnavailable()
    return JSONResponse(status_code=error.status_code, content=error_payload(error))


@app.exception_handler(RequestValidationError)
async def fastapi_request_validation_exception_handler(
    _: Request, exc: RequestValidationError
) -> JSONResponse:
    payload = get_fastapi_exception_payload(exc)
    structlog.contextvars.bind_contextvars(error_code="request_validation_error")
    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    return JSONResponse(status_code=status_code, content=payload)


@app.exception_handler(ServerError)
async def server_error_handler(_: Request, exc: ServerError) -> JSONResponse:
    structlog.contextvars.bind_contextvars(error_code=exc.error_code)
    return JSONResponse(status_code=exc.status_code, content=error_payload(exc))


@app.exception_handler(ClientError)
async def client_error_handler(_: Request, exc: ClientError) -> JSONResponse:
    structlog.contextvars.bind_contextvars(error_code=exc.error_code)
    return JSONResponse(status_code=exc.status_code, content=error_payload(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app", host="0.0.0.0", port=8000, reload=False, access_log=False
    )
