from typing import Any

import pytest
from openai import RateLimitError as OpenAIRateLimitError
from pytest_mock import MockerFixture

from app.adapters.text_embedder import LangChainEmbedderBase
from app.domain.models import Embedding
from app.exceptions import EmbeddingError, RateLimitError


def _make_embedder(mocker: MockerFixture) -> LangChainEmbedderBase:
    obj = object.__new__(LangChainEmbedderBase)
    obj._embedder = mocker.MagicMock()
    obj.model_id = "test-model"
    return obj


def test_embed_texts_success(mocker: MockerFixture):
    embedder = _make_embedder(mocker)
    embed_mock: Any = embedder._embedder.embed_documents
    embed_mock.return_value = [[0.1, 0.2]]

    result = embedder.embed_texts(["hello"])

    assert len(result) == 1
    assert isinstance(result[0], Embedding)
    assert result[0].vector == (0.1, 0.2)
    assert result[0].model_id == "test-model"


def test_embed_texts_rate_limit(mocker: MockerFixture):
    embedder = _make_embedder(mocker)
    embed_mock: Any = embedder._embedder.embed_documents
    embed_mock.side_effect = OpenAIRateLimitError(
        message="rate limited",
        response=mocker.MagicMock(status_code=429, headers={}),
        body=None,
    )

    with pytest.raises(RateLimitError):
        embedder.embed_texts(["hello"])


def test_embed_texts_generic_error(mocker: MockerFixture):
    embedder = _make_embedder(mocker)
    embed_mock: Any = embedder._embedder.embed_documents
    embed_mock.side_effect = ConnectionError("timeout")

    with pytest.raises(EmbeddingError, match="External api error"):
        embedder.embed_texts(["hello"])


def test_embed_query_success(mocker: MockerFixture):
    embedder = _make_embedder(mocker)
    query_mock: Any = embedder._embedder.embed_query
    query_mock.return_value = [0.3, 0.4]

    result = embedder.embed_query("hello")

    assert isinstance(result, Embedding)
    assert result.vector == (0.3, 0.4)
    assert result.model_id == "test-model"


def test_embed_query_rate_limit(mocker: MockerFixture):
    embedder = _make_embedder(mocker)
    query_mock: Any = embedder._embedder.embed_query
    query_mock.side_effect = OpenAIRateLimitError(
        message="rate limited",
        response=mocker.MagicMock(status_code=429, headers={}),
        body=None,
    )

    with pytest.raises(RateLimitError):
        embedder.embed_query("hello")


def test_embed_query_generic_error(mocker: MockerFixture):
    embedder = _make_embedder(mocker)
    query_mock: Any = embedder._embedder.embed_query
    query_mock.side_effect = RuntimeError("api down")

    with pytest.raises(EmbeddingError, match="External api error"):
        embedder.embed_query("hello")
