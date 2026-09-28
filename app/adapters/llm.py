from collections.abc import Sequence
from typing import Any

import structlog
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai.chat_models import ChatOpenAI
from openai import RateLimitError as OpenAIRateLimitError
from pydantic import SecretStr

from app.exceptions import LLMError, RateLimitError

logger = structlog.get_logger(__name__)


class LangChainOpenAILLMChat:
    def __init__(
        self,
        model_id: str,
        base_url: str,
        api_key: str,
        max_retries: int,
        system_prompt: str = "",
        callbacks: Sequence[Any] | None = None,
    ) -> None:
        self.model_id = model_id
        self.system_prompt = system_prompt
        self._callbacks = list(callbacks) if callbacks else []

        model = ChatOpenAI(
            model=model_id,
            base_url=base_url,
            api_key=SecretStr(api_key),
            max_retries=max_retries,
        )
        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("user", "{input}")]
        )
        self._chain = prompt | model | StrOutputParser()

    def invoke(self, query: str) -> str:
        try:
            config: RunnableConfig | None = None
            if self._callbacks:
                config = {"callbacks": self._callbacks}
            return self._chain.invoke({"input": query}, config=config)

        except OpenAIRateLimitError as e:
            logger.warning(
                "llm_rate_limit_exceeded",
                operation="invoke",
                model=self.model_id,
                query_count=len(query),
            )
            raise RateLimitError() from e

        except Exception as e:
            logger.exception(
                "llm_invocation_failed",
                operation="invoke",
                model=self.model_id,
                query_count=len(query),
            )
            raise LLMError() from e


class LangchainPromptTemplate:
    def __init__(self, template: str) -> None:
        self.template = ChatPromptTemplate.from_template(template)

    def format(self, **kwargs: str) -> str:
        return self.template.format(**kwargs)
