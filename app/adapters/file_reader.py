from liteparse import LiteParse

from app.exceptions import NoExtractableText, UnsupportedFileType
from app.services.ports import FileReader

_PDF_MAGIC = b"%PDF-"


class PdfReader:
    def validate(self, content: bytes, header: bytes) -> None:
        if not content:
            raise UnsupportedFileType("Empty file")
        if not header.startswith(_PDF_MAGIC):
            raise UnsupportedFileType("Only PDF files are supported")

    def read(self, content: bytes) -> str:
        self.validate(content, content[:5])

        parser = LiteParse(ocr_enabled=True)
        result = parser.parse(content)

        all_texts = []
        for page in result.pages:
            full_text = "\n".join([item.text for item in page.text_items])
            all_texts.append(full_text)

        text = "\n\n".join(all_texts)
        if not text.strip():
            raise NoExtractableText("No text could be extracted from this PDF")

        return text


class MarkdownReader:
    def validate(self, content: bytes, header: bytes) -> None:
        _ = header
        if not content:
            raise UnsupportedFileType("Empty file")

    def read(self, content: bytes) -> str:
        self.validate(content, b"")
        text = content.decode("utf-8")
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
