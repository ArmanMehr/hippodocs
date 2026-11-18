from collections.abc import Sequence
from typing import Any

import structlog
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langchain_openai.chat_models import ChatOpenAI
from openai import APITimeoutError
from openai import RateLimitError as OpenAIRateLimitError
from pydantic import SecretStr

from app import get_settings
from app.domain.models import Content
from app.exceptions import LLMError, LLMTimeoutError, RateLimitError

logger = structlog.get_logger(__name__)


class LangChainOpenAILLMChat:
    def __init__(
        self,
        model_id: str,
        base_url: str,
        api_key: str,
        max_retries: int | None = None,
        system_prompt: str = "",
        callbacks: Sequence[Any] | None = None,
    ) -> None:
        self.model_id = model_id
        self.system_prompt = system_prompt
        self._callbacks = list(callbacks) if callbacks else []

        base_url_arg = base_url if base_url else None
        max_retries = (
            max_retries if max_retries is not None else get_settings().LLM_MAX_RETRIES
        )
        model = ChatOpenAI(
            model=model_id,
            base_url=base_url_arg,
            api_key=SecretStr(api_key),
            max_retries=max_retries,
        )

        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("user", "{input}")]
        )
        self._chain = prompt | model | StrOutputParser()

    def invoke(self, query: Content) -> Content:
        try:
            config: RunnableConfig | None = None
            if self._callbacks:
                config = {"callbacks": self._callbacks}
            return Content(self._chain.invoke({"input": query.value}, config=config))  # type: ignore

        except OpenAIRateLimitError as e:
            raise RateLimitError() from e

        except APITimeoutError as e:
            raise LLMTimeoutError() from e

        except Exception as e:
            raise LLMError() from e


class LangChainOllamaLLMChat:
    def __init__(
        self,
        model_id: str,
        base_url: str,
        system_prompt: str = "",
        callbacks: Sequence[Any] | None = None,
    ) -> None:
        self.model_id = model_id
        self.system_prompt = system_prompt
        self._callbacks = list(callbacks) if callbacks else []

        base_url_arg = base_url if base_url else None
        model = ChatOllama(model=model_id, base_url=base_url_arg)
        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("user", "{input}")]
        )
        self._chain = prompt | model | StrOutputParser()

    def invoke(self, query: Content) -> Content:
        try:
            config: RunnableConfig | None = None
            if self._callbacks:
                config = {"callbacks": self._callbacks}
            return Content(self._chain.invoke({"input": query.value}, config=config))  # type: ignore

        except Exception as e:
            raise LLMError() from e


class LangchainPromptTemplate:
    def __init__(self, template: str) -> None:
        self.template = ChatPromptTemplate.from_template(template)

    def format(self, **kwargs: str) -> Content:
        return Content(self.template.format(**kwargs))
