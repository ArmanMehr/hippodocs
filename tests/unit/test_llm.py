from typing import Any, cast

from app.adapters.llm import LangChainOpenAILLMChat
from app.domain.models import Content


class RecordingChain:
    def __init__(self) -> None:
        self.inputs: dict[str, str] | None = None

    def invoke(self, inputs: dict[str, str], config: object = None) -> str:
        self.inputs = inputs
        return "Answer"


def test_openai_adapter_passes_content_value_to_chain() -> None:
    chain = RecordingChain()

    adapter = object.__new__(LangChainOpenAILLMChat)
    adapter._chain = cast(Any, chain)
    adapter._callbacks = []

    result = adapter.invoke(Content("first line\nsecond line"))

    assert chain.inputs == {"input": "first line\nsecond line"}
    assert result == Content("Answer")
