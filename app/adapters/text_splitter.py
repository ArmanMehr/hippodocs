from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.domain.models import Content


class LangChainRecursiveTextSplitter:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self._chunker = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def split_text(self, text: str) -> list[Content]:
        return [
            Content(value=part)
            for part in self._chunker.split_text(text=text)
            if part.strip()
        ]
