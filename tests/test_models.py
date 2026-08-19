import pytest

from app.domain.models import (
    Chunk,
    Content,
    Document,
    Embedding,
    EmptyContentError,
    EmptyVectorError,
    Workspace,
)


def test_empty_content():
    with pytest.raises(EmptyContentError):
        Content("")

    with pytest.raises(EmptyContentError):
        Content("  ")


def test_no_embedding_chunk():
    chunk = Chunk(document_id=1, content=Content("text"))

    assert chunk.content == Content("text")
    assert chunk.document_id == 1
    assert not chunk.has_embedding()
    assert chunk.embedding_vector is None
    assert chunk.embedding_model_id is None


def test_empty_embedding_raises_error():
    with pytest.raises(EmptyVectorError):
        Embedding(())


def test_add_embedding_chunk():
    chunk = Chunk(document_id=1, content=Content("text"))
    chunk.add_embedding(Embedding((1.0, 2.0, 3.0)))

    assert chunk.has_embedding()
    assert chunk.embedding_vector == [1.0, 2.0, 3.0]


def test_document_no_chunks():
    ws = Workspace(name="ws")
    doc = Document(content=Content("Text"), workspace=ws, title="doc")
    assert not doc.is_preprocessed
    doc.mark_preprocessed()
    assert doc.is_preprocessed
