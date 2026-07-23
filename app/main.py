import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from app import setup_logging
from app.configs import get_settings
from app.database import SessionLocal
from app.limiter import limiter
from app.routers.ask import router as ask_router
from app.routers.documents import router as documents_router
from app.services.factory import (
    create_cache_embeddings_provider,
    create_embedding_service,
    create_ingestion_pipeline,
    create_llm_provider,
    create_rag_service,
)

logger = logging.getLogger(__name__)

try:
    level = getattr(logging, get_settings().LOG_LEVEL)
except Exception:
    level = logging.INFO
setup_logging(level=level)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting the RAG API.")
    cache_embeddings = create_cache_embeddings_provider()
    app.state.llm_provider = create_llm_provider()
    app.state.cache_embeddings = cache_embeddings
    app.state.embedding_service = create_embedding_service(cache_embeddings)
    app.state.ingestion_pipeline = create_ingestion_pipeline(
        embedding_service=app.state.embedding_service
    )
    session = SessionLocal()
    app.state.rag_service = create_rag_service(
        embeddings_model=cache_embeddings,
        session=session,
    )
    try:
        yield
    finally:
        session.close()


app = FastAPI(lifespan=lifespan, title="Simple RAG API", version="1.0.0")
app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore


app.include_router(documents_router)
app.include_router(ask_router)


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    if get_settings().ENV == "dev":
        import uvicorn

        uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
