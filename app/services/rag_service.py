from collections.abc import Sequence
from logging import getLogger

from app.adapters.llm import LangchainPromptTemplate
from app.domain.models import Chunk, Content, Document, Workspace
from app.exceptions import DocumentProcessingError, WorkspaceNotFound
from app.services.ports import (
    FileReader,
    LLMChat,
    PromptTemplate,
    TextEmbedder,
    TextSplitter,
)
from app.services.uow import UnitOfWork

logger = getLogger(__name__)


DEFAULT_RAG_PROMPT_TEMPLATE = LangchainPromptTemplate(
    """
    Answer the question based only on the following context:
    {context}
    Question: {question}
    Answer concisely. If unsure, state 'I don't know.'
    """
)


class WorkspaceService:
    def __init__(self, uow: UnitOfWork) -> None:
        self.uow = uow

    def create_workspace(self, name: str) -> int:
        return self.new_workspace(name)

    def new_workspace(self, name: str) -> int:
        with self.uow:
            workspace = Workspace(name=name)
            self.uow.workspaces.add(workspace)
            self.uow.commit()
            workspace_id = workspace.workspace_id  # type: ignore[attr-defined]
        return workspace_id

    def new_document(
        self, workspace_id: int, title: str | None, text: str
    ) -> int | None:
        with self.uow:
            workspace = self.uow.workspaces.get(workspace_id)
            if workspace is None:
                raise WorkspaceNotFound(workspace_id)

            document = Document(content=Content(text), workspace=workspace, title=title)
            self.uow.documents.add(document)
            self.uow.commit()
            document_id = document.document_id  # type: ignore[attr-defined]
        return document_id

    def delete_document(self, document_id: int) -> None:
        with self.uow:
            self.uow.documents.delete(document_id)
            self.uow.commit()

    def delete_workspace(self, workspace_id: int) -> None:
        with self.uow:
            self.uow.workspaces.delete(workspace_id)
            self.uow.commit()

    def get_workspace(self, workspace_id: int) -> Workspace | None:
        with self.uow:
            return self.uow.workspaces.get(workspace_id)

    def list_workspaces(self, skip: int, limit: int) -> tuple[list[Workspace], int]:
        return self.get_workspaces(skip=skip, limit=limit)

    def get_workspaces(self, skip: int, limit: int) -> tuple[list[Workspace], int]:
        with self.uow:
            workspaces, total = self.uow.workspaces.get_all(skip=skip, limit=limit)
        return workspaces, total


class DocumentIngestionService:
    def __init__(
        self, uow: UnitOfWork, splitter: TextSplitter, embedder: TextEmbedder
    ) -> None:
        self.uow = uow
        self.splitter = splitter
        self.embedder = embedder

    def add_document(
        self, reader: FileReader, file_data: bytes, workspace_id: int, title: str | None
    ) -> int:
        text = reader.read(file_data)
        with self.uow:
            workspace = self.uow.workspaces.get(workspace_id)
            if workspace is None:
                raise WorkspaceNotFound(workspace_id)

            document = Document(content=Content(text), workspace=workspace, title=title)
            self.uow.documents.add(document)
            self.uow.commit()
            document_id = document.document_id  # type: ignore[attr-defined]

        self.ingest_document(document_id)
        return document_id

    def ingest_document(self, document_id: int) -> None:
        with self.uow:
            document = self.uow.documents.get(document_id=document_id)
            if document is None:
                raise DocumentProcessingError(document_id)

            contents = self.splitter.split_text(document.content.value)
            if not contents:
                document.mark_preprocessed()
                return

            chunks = self._embed_and_chunk_contents(document.document_id, contents)  # type: ignore[attr-defined]
            document.mark_preprocessed()

            self.uow.chunks.save_all(chunks)
            self.uow.commit()

    def ingest_workspace(self, workspace_id: int) -> None:
        with self.uow:
            if self.uow.workspaces.get(workspace_id) is None:
                raise WorkspaceNotFound(workspace_id)

            documents: Sequence[Document] = (
                self.uow.documents.list_unpreprocessed_by_workspace(workspace_id)
            )

            all_chunks: list[Chunk] = []
            for document in documents:
                contents = self.splitter.split_text(document.content.value)
                if not contents:
                    document.mark_preprocessed()
                    continue

                chunks = self._embed_and_chunk_contents(document.document_id, contents)  # type: ignore[attr-defined]
                document.mark_preprocessed()
                all_chunks.extend(chunks)

            self.uow.chunks.save_all(all_chunks)
            self.uow.commit()

    def _embed_and_chunk_contents(self, document_id: int, contents: Sequence[Content]):
        embeddings = self.embedder.embed_texts([c.value for c in contents])
        return [
            Chunk(document_id=document_id, content=content).add_embedding(embedding)
            for content, embedding in zip(contents, embeddings, strict=True)
        ]


class RagService:
    def __init__(
        self,
        uow: UnitOfWork,
        embedder: TextEmbedder,
        llm: LLMChat,
        top_k: int,
        chat_prompt: PromptTemplate = DEFAULT_RAG_PROMPT_TEMPLATE,
    ) -> None:
        self.uow = uow
        self.embedder = embedder
        self.llm = llm
        self.top_k = top_k
        self.chat_prompt = chat_prompt

    def _retrieve(self, workspace_id: int, query_text: str) -> str:
        with self.uow:
            workspace = self.uow.workspaces.get(workspace_id)
            if workspace is None:
                raise WorkspaceNotFound(workspace_id)
            query_embedding = self.embedder.embed_query(query_text)
            found_chunks = self.uow.chunks.find_similar_in_workspace(
                workspace_id=workspace_id,
                query_vector=list(query_embedding.vector),
                top_k=self.top_k,
            )
            return "\n".join([chunk.content.value for chunk in found_chunks])

    def query(self, workspace_id: int, query_text: str) -> str:
        context = self._retrieve(workspace_id, query_text)
        prompt_text = self.chat_prompt.format(context=context, question=query_text)
        return self.llm.invoke(prompt_text)
