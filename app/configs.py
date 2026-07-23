from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Configs(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="allow", frozen=True)

    ENV: Literal["dev", "prd"] = "dev"
    LOG_LEVEL: str = "DEBUG"

    # PostgreSQL Database
    DATABASE_URL: str = ""

    # Embedding
    EMBEDDING_PROVIDER: str = "ollama"
    EMBEDDING_BASE_URL: str = "http://localhost:11434"
    EMBEDDING_MODEL: str = ""
    EMBEDDING_FALLBACK_PROVIDER: str = ""
    EMBEDDING_FALLBACK_BASE_URL: str = ""
    EMBEDDING_FALLBACK_MODEL: str = ""
    DIMENSIONS: int = 384

    # LLM
    LLM_PROVIDER: str = "openai"
    LLM_BASE_URL: str = "http://localhost:3001/v1"
    LLM_MODEL: str = ""
    LLM_FALLBACK_PROVIDER: str = ""
    LLM_FALLBACK_BASE_URL: str = ""
    LLM_FALLBACK_MODEL: str = ""
    LLM_MAX_RETRIES: int = 3
    OPENAI_API_KEY: str = "no-key"

    # RAG Settings
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    TOP_K: int = 25

    # API
    MAX_FILESIZE: int = 10 * 1024 * 1024


@lru_cache
def get_settings():
    return Configs()
