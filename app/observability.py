import os
from enum import Enum, auto

import structlog
from langfuse import Langfuse, observe
from langfuse.langchain import CallbackHandler

logger = structlog.get_logger(__name__)


class LangfuseState(Enum):
    UNINITIALIZED = auto()
    DISABLED = auto()
    READY = auto()
    FAILED = auto()


_state: LangfuseState = LangfuseState.UNINITIALIZED
_client: Langfuse | None = None


def initialize_langfuse() -> None:
    global _state, _client

    if _state is not LangfuseState.UNINITIALIZED:
        return

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")

    if not public_key or not secret_key:
        logger.warning("langfuse_not_configured")
        _state = LangfuseState.DISABLED
        return

    try:
        _client = Langfuse(public_key=public_key, secret_key=secret_key)
        _state = LangfuseState.READY
    except Exception:
        logger.exception("langfuse_initialization_failed")
        _state = LangfuseState.FAILED


def get_langfuse_client() -> Langfuse | None:
    initialize_langfuse()
    return _client


def flush_langfuse() -> None:
    client = get_langfuse_client()
    if client is not None:
        try:
            client.flush()
        except Exception:
            logger.exception("langfuse_flush_failed", operation="flush_langfuse")


def is_langfuse_enabled() -> bool:
    return get_langfuse_client() is not None


def create_langchain_callback_handler():
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
