from typing import Any

import pytest
from pytest_mock import MockerFixture

from app.adapters.file_reader import FileReaderRegistry, MarkdownReader, PdfReader
from app.exceptions import NoExtractableText, UnsupportedFileType


def test_pdf_validate_empty_file():
    with pytest.raises(UnsupportedFileType, match="Empty file"):
        PdfReader().validate(b"", b"")


def test_pdf_validate_non_pdf_header():
    with pytest.raises(UnsupportedFileType, match="Only PDF"):
        PdfReader().validate(b"content", b"PNG-")


def test_pdf_validate_valid_header():
    PdfReader().validate(b"content", b"%PDF-1.4")


def test_pdf_read_extracts_text(mocker: MockerFixture):
    page = mocker.MagicMock()
    text_items: Any = [mocker.MagicMock(text="Hello"), mocker.MagicMock(text="World")]
    page.text_items = text_items
    result = mocker.MagicMock()
    result.pages = [page]

    mock_parse = mocker.patch("app.adapters.file_reader.LiteParse")
    mock_parse.return_value.parse.return_value = result

    text = PdfReader().read(b"%PDF-1.4 fake content")
    assert text == "Hello\nWorld"


def test_pdf_read_no_extractable_text(mocker: MockerFixture):
    page = mocker.MagicMock()
    page.text_items = [mocker.MagicMock(text="")]
    result = mocker.MagicMock()
    result.pages = [page]

    mock_parse = mocker.patch("app.adapters.file_reader.LiteParse")
    mock_parse.return_value.parse.return_value = result

    with pytest.raises(NoExtractableText):
        PdfReader().read(b"%PDF-1.4 fake content")


def test_markdown_validate_empty_file():
    with pytest.raises(UnsupportedFileType, match="Empty file"):
        MarkdownReader().validate(b"", b"")


def test_markdown_read_content():
    assert MarkdownReader().read(b"# Hello") == "# Hello"


def test_markdown_read_whitespace_only():
    with pytest.raises(NoExtractableText):
        MarkdownReader().read(b"   \n\n   ")


def test_registry_register_and_get():
    reg = FileReaderRegistry()
    reader = MarkdownReader()
    reg.register(".md", reader)
    assert reg.get(".md") is reader


def test_registry_get_unregistered():
    with pytest.raises(UnsupportedFileType):
        FileReaderRegistry().get(".xyz")


def test_registry_supported_extensions():
    reg = FileReaderRegistry()
    reg.register(".md", MarkdownReader())
    reg.register(".pdf", PdfReader())
    assert reg.supported_extensions == {".md", ".pdf"}
