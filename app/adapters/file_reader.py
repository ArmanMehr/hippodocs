import time

from liteparse import LiteParse

from app import structlog
from app.exceptions import (
    FileProcessingError,
    NoExtractableText,
    UnsupportedFileType,
)
from app.observability import observe
from app.services.ports import FileReader

logger = structlog.get_logger(__name__)


class PdfReader:
    def validate(self, content: bytes, header: bytes) -> None:
        if not content:
            raise UnsupportedFileType("Empty file")
        if not header.startswith(b"%PDF-"):
            raise UnsupportedFileType("Only PDF files are supported")

    @observe(name="read-pdf", as_type="chain")
    def read(self, content: bytes) -> str:
        start_time = time.time()
        self.validate(content, content[:5])
        try:
            parser = LiteParse(ocr_enabled=True)
            result = parser.parse(content)

            all_texts = []
            for page in result.pages:
                full_text = "\n".join([item.text for item in page.text_items])
                all_texts.append(full_text)

            text = "\n\n".join(all_texts)

        except Exception as e:
            logger.exception(
                "file_processing_failed",
                operation="pdf_read",
                duration_ms=round((time.perf_counter() - start_time) * 1000, 2),
            )
            raise FileProcessingError() from e

        if not text.strip():
            raise NoExtractableText("No text could be extracted from this PDF file")

        return text


class MarkdownReader:
    def validate(self, content: bytes, header: bytes) -> None:
        _ = header
        if not content:
            raise UnsupportedFileType("Empty file")

    @observe(name="read-markdown", as_type="chain")
    def read(self, content: bytes) -> str:
        start_time = time.time()
        self.validate(content, b"")
        try:
            text = content.decode("utf-8")
        except Exception as e:
            logger.exception(
                "file_processing_failed",
                operation="markdown_read",
                duration_ms=round((time.perf_counter() - start_time) * 1000, 2),
            )
            raise FileProcessingError() from e

        if not text.strip():
            raise NoExtractableText(
                "No text could be extracted from this Markdown file"
            )
        return text


class FileReaderRegistry:
    def __init__(self) -> None:
        self._readers: dict[str, FileReader] = {}

    def register(self, extension: str, reader: FileReader) -> None:
        self._readers[extension] = reader

    def get(self, extension: str) -> FileReader:
        reader = self._readers.get(extension)
        if reader is None:
            raise UnsupportedFileType(f"Unsupported file type: {extension}")
        return reader

    @property
    def supported_extensions(self) -> set[str]:
        return set(self._readers.keys())
