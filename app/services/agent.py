from collections.abc import Sequence
from logging import getLogger
from typing import ClassVar

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableParallel,
    RunnablePassthrough,
)
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Document as DocumentModel

logger = getLogger(__name__)


class RagService:
    DEFAULT_PROMPT: ClassVar[ChatPromptTemplate] = ChatPromptTemplate.from_template(
        "Answer the question based only on the following context:\n\n"
        "{context}\n\n"
        "Question: {question}\n\n"
        "Answer concisely. If unsure, state 'I don't know.'"
    )

    def __init__(
        self,
        db_session: Session,
        embeddings_model: Embeddings,
        llm: BaseChatModel,
        top_k: int,
        chat_prompt: ChatPromptTemplate | None = None,
    ) -> None:
        self._db_session = db_session
        self._embeddings_model = embeddings_model
        self._llm = llm
        self._top_k = top_k
        self._chat_prompt = chat_prompt or self.DEFAULT_PROMPT

        self._rag_pipeline: Runnable[str, str] = (
            RunnableParallel(
                {
                    "context": self.retrieve,
                    "question": RunnablePassthrough(),
                }
            )
            | self._chat_prompt
            | self._llm
            | StrOutputParser()
        )

    def retrieve(self, query_text: str) -> str:
        query_vector: list[float] = self._embeddings_model.embed_query(query_text)

        stmt = (
            select(DocumentModel.content)
            .order_by(DocumentModel.embedding.cosine_distance(query_vector))
            .limit(self._top_k)
        )

        documents: Sequence[str] = self._db_session.scalars(stmt).all()
        return "\n\n".join(documents)

    def query(self, query_text: str) -> str:
        return self._rag_pipeline.invoke(query_text)
