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


def make_chunk() -> Chunk:
    return Chunk(document_id=1, content=Content("text"))


def make_document() -> Document:
    return Document(
        content=Content("Text"), workspace=Workspace(name="ws"), title="doc"
    )


@pytest.mark.parametrize("text", ["", "  "])
def test_empty_content(text: str):
    with pytest.raises(EmptyContentError):
        Content(text)


def test_empty_embedding_raises_error():
    with pytest.raises(EmptyVectorError):
        Embedding(())


def test_no_embedding_chunk():
    chunk = make_chunk()

    assert chunk.content == Content("text")
    assert chunk.document_id == 1
    assert not chunk.has_embedding()
    assert chunk.embedding_vector is None
    assert chunk.embedding_model_id is None


def test_add_embedding_chunk():
    chunk = make_chunk()
    chunk.add_embedding(Embedding((1.0, 2.0, 3.0)))

    assert chunk.has_embedding()
    assert chunk.embedding_vector == [1.0, 2.0, 3.0]


def test_set_embedding_chunk():
    chunk = make_chunk()
    chunk.embedding = Embedding((1.0, 2.0, 3.0))

    assert chunk.has_embedding()
    assert chunk.embedding_vector == [1.0, 2.0, 3.0]


def test_zero_dim_empty_embedding_chunk():
    assert not make_chunk().has_embedding()
    assert make_chunk().ndim == 0


def test_document_mark_preprocessed():
    doc = make_document()

    assert not doc.is_preprocessed
    doc.mark_preprocessed()
    assert doc.is_preprocessed


def test_chunk_embedding_property_returns_embedding():
    chunk = make_chunk()
    chunk.embedding = Embedding((1.0, 2.0), model_id="m1")

    emb = chunk.embedding
    assert isinstance(emb, Embedding)
    assert emb.vector == (1.0, 2.0)
    assert emb.model_id == "m1"


def test_chunk_embedding_setter_none_clears():
    chunk = make_chunk()
    chunk.embedding = Embedding((1.0,))
    chunk.embedding = None

    assert chunk.embedding_vector is None
    assert chunk.embedding_model_id is None
    assert not chunk.has_embedding()


def test_chunk_ndim_with_vector():
    chunk = make_chunk()
    chunk.embedding = Embedding((1.0, 2.0, 3.0))
    assert chunk.ndim == 3
