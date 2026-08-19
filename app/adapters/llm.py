from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI


class LangChainOpenAILLMChat:
    def __init__(
        self, model_id: str, base_url: str, api_key: str, max_retries: int
    ) -> None:
        self.model_id = model_id

        model = ChatOpenAI(
            model=model_id, base_url=base_url, api_key=api_key, max_retries=max_retries
        )
        self._chain = model | StrOutputParser()

    def invoke(self, query: str) -> str:
        return self._chain.invoke(query)


class LangchainPromptTemplate:
    def __init__(self, template: str) -> None:
        self.template = ChatPromptTemplate.from_template(template)

    def format(self, **kwargs: str) -> str:
        return self.template.format(**kwargs)
