from typing import Any
from unittest.mock import patch

import pytest
from openai import RateLimitError as OpenAIRateLimitError
from pytest_mock import MockerFixture

from app.adapters.llm import LangChainOpenAILLMChat
from app.exceptions import LLMError, RateLimitError


def _make_llm_chat(mocker: MockerFixture) -> LangChainOpenAILLMChat:
    with patch("app.adapters.llm.ChatOpenAI"):
        obj = LangChainOpenAILLMChat(
            model_id="test-model",
            base_url="http://localhost",
            api_key="key",
            max_retries=0,
        )
    obj._chain = mocker.MagicMock()
    return obj


def test_invoke_rate_limit(mocker: MockerFixture):
    llm_chat = _make_llm_chat(mocker)
    chain_mock: Any = llm_chat._chain.invoke
    chain_mock.side_effect = OpenAIRateLimitError(
        message="rate limited",
        response=mocker.MagicMock(status_code=429, headers={}),
        body=None,
    )

    with pytest.raises(RateLimitError):
        llm_chat.invoke("Hi")


def test_invoke_generic_error(mocker: MockerFixture):
    llm_chat = _make_llm_chat(mocker)
    chain_mock: Any = llm_chat._chain.invoke
    chain_mock.side_effect = ConnectionError("timeout")

    with pytest.raises(LLMError, match="External api error"):
        llm_chat.invoke("Hi")
