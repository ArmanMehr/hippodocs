import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from openai import RateLimitError as OpenAIRateLimitError

from app.exceptions import LLMError, RateLimitError

logger = logging.getLogger(__name__)


class LangChainOpenAILLMChat:
    def __init__(
        self,
        model_id: str,
        base_url: str,
        api_key: str,
        max_retries: int,
        system_prompt: str = "",
    ) -> None:
        self.model_id = model_id
        self.system_prompt = system_prompt

        model = ChatOpenAI(
            model=model_id, base_url=base_url, api_key=api_key, max_retries=max_retries
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("user", "{input}"),
            ]
        )
        self._chain = prompt | model | StrOutputParser()

    def invoke(self, query: str) -> str:
        try:
            return self._chain.invoke({"input": query})
        except OpenAIRateLimitError as e:
            logger.warning("LLM rate limit exceeded: %s", e)
            raise RateLimitError() from e
        except Exception as e:  # noqa: BLE001
            logger.error("LLM invocation failed: %s", e)
            raise LLMError("External api error")


class LangchainPromptTemplate:
    def __init__(self, template: str) -> None:
        self.template = ChatPromptTemplate.from_template(template)

    def format(self, **kwargs: str) -> str:
        return self.template.format(**kwargs)
