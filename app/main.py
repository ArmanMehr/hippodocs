import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from app import setup_logging
from app.adapters.orm import start_mappers
from app.api.v1 import router as api_v1_router
from app.configs import get_settings
from app.exceptions import AppError, error_payload
from app.limiter import limiter
from app.services.factory import (
    create_file_readers,
    create_ingestion_service,
    create_rag_service,
    create_workspace_service,
)

level = getattr(logging, get_settings().LOG_LEVEL)
setup_logging(level=level)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting the RAG API.")
    start_mappers()
    app.state.ingestion_service = create_ingestion_service()
    app.state.rag_service = create_rag_service()
    app.state.workspace_service = create_workspace_service()
    app.state.file_readers = create_file_readers()
    try:
        yield
    finally:
        logger.info("Stopping the RAG API.")


app = FastAPI(lifespan=lifespan, title="Simple RAG API", version="1.0.0")
app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore


@app.exception_handler(AppError)
async def app_error_handler(_request: Request, exception: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=exception.status_code, content=error_payload(exception)
    )


app.include_router(api_v1_router)


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    if get_settings().ENV == "dev":
        import uvicorn

        uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
