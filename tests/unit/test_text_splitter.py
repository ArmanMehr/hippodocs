from app.adapters.text_splitter import LangChainRecursiveTextSplitter
from app.domain.models import Content


def _make_splitter(
    chunk_size: int = 100, chunk_overlap: int = 0
) -> LangChainRecursiveTextSplitter:
    return LangChainRecursiveTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )


def test_splits_into_chunks():
    splitter = _make_splitter(chunk_size=10)
    result = splitter.split_text("Hello world this is a test")

    assert all(isinstance(c, Content) for c in result)
    assert all(len(c.value) <= 10 for c in result)


def test_filters_blank_chunks():
    splitter = _make_splitter(chunk_size=5)
    result = splitter.split_text("a\n\n\n\nb")

    values = [c.value for c in result]
    assert all(v.strip() for v in values)
    assert any("a" in v for v in values)
    assert any("b" in v for v in values)


def test_empty_text():
    splitter = _make_splitter()
    assert splitter.split_text("") == []


def test_whitespace_only():
    splitter = _make_splitter()
    assert splitter.split_text("   \n\n   ") == []


def test_short_text_no_split():
    splitter = _make_splitter(chunk_size=100)
    result = splitter.split_text("Hello")

    assert len(result) == 1
    assert result[0].value == "Hello"
