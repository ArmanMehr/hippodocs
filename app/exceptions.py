from __future__ import annotations

import datetime
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, override

from fastapi.exceptions import RequestValidationError


def _iso_now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


@dataclass(frozen=True)
class ErrorPayload:
    detail: str
    error_code: str
    timestamp: str = _iso_now()

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


def get_fastapi_exception_payload(exc: RequestValidationError) -> dict[str, Any]:
    message = "Validation errors:"
    for error in exc.errors():
        message += f"\nField: {error['loc']}, Error: {error['msg']}"
    payload = ErrorPayload(detail=message, error_code="request_validation_error")
    return payload.asdict()


class AppError(Exception):
    status_code: int = 500
    error_code: str = "internal_error"
    detail: str | None = None

    def __init__(
        self, detail: str | None = None, *, error_code: str | None = None
    ) -> None:
        self.detail = detail or self.detail or self.__class__.__name__
        if error_code is not None:
            self.error_code = error_code
        super().__init__(self.detail)

    @override
    def __str__(self) -> str:
        return self.detail or super().__str__()


def error_payload(exc: AppError) -> Mapping[str, Any]:
    return ErrorPayload(detail=exc.detail or "", error_code=exc.error_code).asdict()


class WorkspaceNotFound(AppError):
    status_code = 404
    error_code = "workspace_not_found"
    detail = "Workspace not found"


class DocumentNotFound(AppError):
    status_code = 404
    error_code = "document_not_found"
    detail = "Document not found"


class FileProcessingError(AppError):
    status_code = 503
    error_code = "file_processing_error"
    detail = "File could not be processed"


class DatabaseUnavailable(AppError):
    status_code = 503
    error_code = "database_unavailable"
    detail = "Database is unavailable"


class ProviderConnectionError(AppError):
    status_code = 503
    error_code = "provider_unavailable"


class RateLimitError(AppError):
    status_code = 429
    error_code = "rate_limit_exceeded"


class ValidationError(AppError):
    status_code = 422
    error_code = "validation_error"


class SuspiciousInputError(ValidationError):
    error_code = "suspicious_input"
    detail = "Suspicious input detected"


class UnsupportedFileType(ValidationError):
    error_code = "unsupported_file_type"


class NoExtractableText(ValidationError):
    error_code = "no_extractable_text"


class FileTooLarge(ValidationError):
    status_code = 413
    error_code = "file_too_large"


class MissingFilename(ValidationError):
    status_code = 422
    error_code = "missing_filename"


class ExternalAPIError(AppError):
    status_code = 502
    error_code = "external_api_error"
    detail = "External api error"


class EmbeddingError(ExternalAPIError):
    status_code = 502
    error_code = "embedding_error"


class EmbeddingTimeoutError(EmbeddingError):
    status_code = 504
    error_code = "embedding_timeout"
    detail = "Embedding request timed out"


class LLMError(ExternalAPIError):
    status_code = 502
    error_code = "llm_error"


class LLMTimeoutError(LLMError):
    status_code = 504
    error_code = "llm_timeout"
    detail = "LLM request timed out"
