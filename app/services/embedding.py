from typing import Optional, Sequence

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import TextSplitter

from app.models import Document as DocumentModel
from app.services.repository import DocumentRepository


class ExactMatchDeduplicator:
    def __init__(self, repository: DocumentRepository) -> None:
        self._repository = repository

    def filter_new(self, documents: Sequence[Document]) -> list[Document]:
        if not documents:
            return []
        contents = [doc.page_content for doc in documents]
        existing = self._repository.get_existing_contents(contents)
        return [doc for doc in documents if doc.page_content.lower() not in existing]


class DocumentEmbeddingService:
    def __init__(
        self,
        splitter: TextSplitter,
        embeddings_model: Embeddings,
    ) -> None:
        self._splitter = splitter
        self._embeddings_model = embeddings_model

    def split_documents(self, documents: Sequence[Document]) -> list[Document]:
        return self._splitter.split_documents(documents)

    def embed_documents(self, documents: Sequence[Document]) -> list[DocumentModel]:
        documents = [doc for doc in documents if doc.page_content.strip()]
        if not documents:
            return []

        texts = [doc.page_content for doc in documents]
        vectors = self._embeddings_model.embed_documents(texts)

        return [
            DocumentModel(content=doc.page_content, embedding=vector)
            for doc, vector in zip(documents, vectors)
        ]


class DocumentIngestionPipeline:
    def __init__(
        self,
        embedding_service: DocumentEmbeddingService,
        repository: DocumentRepository,
        deduplicator: Optional[ExactMatchDeduplicator] = None,
    ) -> None:
        self._embedding_service = embedding_service
        self._repository = repository
        self._deduplicator = deduplicator

    def run(self, documents: Sequence[Document]) -> list[DocumentModel]:
        if not documents:
            return []

        split_docs = self._embedding_service.split_documents(documents)
        if not split_docs:
            return []

        new_docs = (
            self._deduplicator.filter_new(split_docs)
            if self._deduplicator
            else split_docs
        )
        if not new_docs:
            return []

        models = self._embedding_service.embed_documents(new_docs)
        self._repository.save_all(models)
        return models
