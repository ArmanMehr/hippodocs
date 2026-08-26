from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, override


class AppError(Exception):
    status_code: int = 500
    error_code: str | None = None
    detail: str | None = None

    def __init__(
        self, detail: str | None = None, *, error_code: str | None = None
    ) -> None:
        self.detail = detail or self.__class__.__name__
        if error_code is not None:
            self.error_code = error_code
        super().__init__(self.detail)

    @override
    def __str__(self) -> str:
        return self.detail or super().__str__()


def _iso_now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def error_payload(exc: Exception) -> Mapping[str, Any]:
    payload: dict[str, Any] = {
        "detail": str(exc),
        "timestamp": _iso_now(),
    }
    if isinstance(exc, AppError) and exc.error_code:
        payload["error_code"] = exc.error_code
    return payload


class WorkspaceNotFound(AppError):
    status_code = 404
    error_code = "workspace_not_found"

    def __init__(self, workspace_id: int) -> None:
        super().__init__(f"Workspace {workspace_id} not found")


class DocumentNotFound(AppError):
    status_code = 404
    error_code = "document_not_found"


class DocumentProcessingError(AppError):
    status_code = 400
    error_code = "document_processing_error"

    def __init__(self, document_id: int) -> None:
        super().__init__(f"Document {document_id} could not be processed")


class RateLimitError(AppError):
    status_code = 429
    error_code = "rate_limit_exceeded"


class ValidationError(AppError):
    status_code = 422
    error_code = "validation_error"


class UnsupportedFileType(ValidationError):
    error_code = "unsupported_file_type"


class NoExtractableText(ValidationError):
    error_code = "no_extractable_text"


class FileTooLarge(ValidationError):
    error_code = "file_too_large"


class MissingFilename(ValidationError):
    error_code = "missing_filename"
