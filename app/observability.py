import logging
import os

from langfuse import Langfuse, get_client, observe
from langfuse.langchain import CallbackHandler

from app.configs import get_settings

logger = logging.getLogger(__name__)


_SDK_INITIALIZED = False


def initialize_langfuse() -> None:
    global _SDK_INITIALIZED
    if _SDK_INITIALIZED:
        return

    settings = get_settings()
    public_key = settings.LANGFUSE_PUBLIC_KEY
    secret_key = settings.LANGFUSE_SECRET_KEY
    base_url = settings.LANGFUSE_BASE_URL or settings.LANGFUSE_HOST

    if not public_key or not secret_key or not base_url:
        logger.warning("Langfuse credentials not configured; tracing disabled")
        return

    # Set environment variables required by Langfuse SDK and OpenTelemetry
    os.environ["LANGFUSE_PUBLIC_KEY"] = public_key
    os.environ["LANGFUSE_SECRET_KEY"] = secret_key
    os.environ["LANGFUSE_HOST"] = base_url
    os.environ["LANGFUSE_BASE_URL"] = base_url
    if getattr(settings, "ENV", None):
        os.environ["LANGFUSE_TRACING_ENVIRONMENT"] = settings.ENV

    try:
        client = get_client()
        if client.auth_check():
            logger.info("Langfuse client initialized and authenticated")
            _SDK_INITIALIZED = True
        else:
            logger.warning("Langfuse authentication failed; tracing disabled")
            _SDK_INITIALIZED = False
            return
    except Exception as e:  # noqa
        logger.error("Failed to initialize Langfuse: %s", e)
        _SDK_INITIALIZED = False
        return


def get_langfuse_client() -> Langfuse | None:
    """Return the singleton :class:`Langfuse` client, or ``None`` if disabled."""
    if not _SDK_INITIALIZED:
        initialize_langfuse()
    if not _SDK_INITIALIZED:
        return None
    return get_client()


def flush_langfuse() -> None:
    """Flush all pending spans/traces to Langfuse (call before app exit)."""
    client = get_langfuse_client()
    if client is not None:
        client.flush()


def is_langfuse_enabled() -> bool:
    """Return ``True`` when a real (authenticated) Langfuse client is available."""
    return get_langfuse_client() is not None


def create_langchain_callback_handler():
    """Build a LangChain ``CallbackHandler`` wired to the global Langfuse client.

    Returns ``None`` when Langfuse is disabled so callers can safely pass it
    into ``config={"callbacks": [...]}``.
    """
    if not is_langfuse_enabled():
        return None

    return CallbackHandler()


__all__ = [
    "Langfuse",
    "create_langchain_callback_handler",
    "flush_langfuse",
    "get_langfuse_client",
    "initialize_langfuse",
    "is_langfuse_enabled",
    "observe",
]
