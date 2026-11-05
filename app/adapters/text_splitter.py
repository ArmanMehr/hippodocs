from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.domain.models import Content
from app.observability import observe


class LangChainRecursiveTextSplitter:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._chunker = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    @observe(
        name="split-text",
        as_type="span",
        capture_input=False,
        capture_output=False,
    )
    def split_text(self, text: str) -> list[Content]:
        chunks = self._chunker.split_text(text=text)
        return [Content(value=part) for part in chunks if part.strip()]
