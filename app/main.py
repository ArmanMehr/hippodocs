from fastapi import FastAPI
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from app import setup_logging
from app.configs import get_settings
from app.limiter import limiter
from app.routers.ask import router as ask_router
from app.routers.documents import router as documents_router

app = FastAPI(title="Simple RAG API", version="1.0.0")
app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore


app.include_router(documents_router)
app.include_router(ask_router)


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import logging

    try:
        level = getattr(logging, get_settings().LOG_LEVEL)
    except Exception:
        level = logging.INFO

    print(level)
    setup_logging(level=level)

    logger = logging.getLogger(__name__)
    logger.info("Starting!")
    if get_settings().ENV == "dev":
        import uvicorn

        uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
