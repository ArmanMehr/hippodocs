from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, override


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


def _iso_now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def error_payload(exc: AppError) -> Mapping[str, Any]:
    return {
        "detail": exc.detail,
        "error_code": exc.error_code,
        "timestamp": _iso_now(),
    }


class WorkspaceNotFound(AppError):
    status_code = 404
    error_code = "workspace_not_found"
    detail = "Workspace not found"


class DocumentNotFound(AppError):
    status_code = 404
    error_code = "document_not_found"
    detail = "Document not found"


# FIX: Add more specific exceptions
class DocumentProcessingError(AppError):
    status_code = 400
    error_code = "document_processing_error"

    def __init__(self) -> None:
        super().__init__("Document could not be processed")


class DatabaseUnavailable(AppError):
    status_code = 503
    error_code = "database_unavailable"
    detail = "Database is unavailable"


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


# TODO: Add more specific exceptions
class EmbeddingError(AppError):
    status_code = 502
    error_code = "embedding_error"
    detail = "External api error"


# TODO: Add more specific exceptions
class LLMError(AppError):
    status_code = 502
    error_code = "llm_error"
    detail = "External api error"
