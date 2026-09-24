from collections.abc import Sequence
from logging import getLogger
from typing import BinaryIO

from app.adapters.file_reader import FileReader, FileReaderRegistry
from app.domain.models import Chunk, Content, Document, Workspace
from app.exceptions import DocumentNotFound, DocumentProcessingError, WorkspaceNotFound
from app.observability import observe
from app.services.ports import (
    InputSanitizer,
    LLMChat,
    OutputValidator,
    PIIRedactor,
    PromptTemplate,
    TextEmbedder,
    TextSplitter,
)
from app.services.uow import UnitOfWork

logger = getLogger(__name__)


class WorkspaceService:
    def __init__(self, uow: UnitOfWork) -> None:
        self.uow = uow

    def create_workspace(self, name: str) -> int:
        return self.new_workspace(name)

    def delete_workspace(self, workspace_id: int) -> None:
        with self.uow:
            self.uow.workspaces.delete(workspace_id)
            self.uow.commit()

    def get_document(self, document_id: int) -> Document:
        with self.uow:
            document = self.uow.documents.get(document_id)
            if document is None:
                raise DocumentNotFound(document_id)
        return document

    def get_workspace(self, workspace_id: int) -> Workspace | None:
        with self.uow:
            return self.uow.workspaces.get(workspace_id)

    def get_workspaces(self, skip: int, limit: int) -> tuple[list[Workspace], int]:
        with self.uow:
            workspaces, total = self.uow.workspaces.get_all(skip=skip, limit=limit)
        return workspaces, total

    def list_documents_in_workspace(
        self, workspace_id: int, skip: int, limit: int
    ) -> tuple[list[Document], int]:
        with self.uow:
            documents, total = self.uow.documents.list_by_workspace(
                workspace_id, skip=skip, limit=limit
            )
        return documents, total

    def list_workspaces(self, skip: int, limit: int) -> tuple[list[Workspace], int]:
        return self.get_workspaces(skip=skip, limit=limit)

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

    def new_workspace(self, name: str) -> int:
        with self.uow:
            workspace = Workspace(name=name)
            self.uow.workspaces.add(workspace)
            self.uow.commit()
            workspace_id = workspace.workspace_id  # type: ignore[attr-defined]
        return workspace_id

    def delete_document(self, workspace_id: int, document_id: int) -> None:
        with self.uow:
            workspace = self.uow.workspaces.get(workspace_id)
            if workspace is None:
                raise WorkspaceNotFound(workspace_id)
            document = self.uow.documents.get(document_id)
            if document is None or document.workspace.workspace_id != workspace_id:  # type: ignore[attr-defined]
                raise DocumentNotFound(document_id)
            self.uow.documents.delete(document_id)
            self.uow.commit()


class FileReaderService:
    def __init__(self, registry: FileReaderRegistry) -> None:
        self.registry = registry

    def read(self, content: BinaryIO, file_name: str) -> str:
        extension = file_name.rsplit(".", 1)[-1].lower()
        reader = self._get_reader(extension)
        return reader.read(content.read())

    def get_filename_no_ext(self, file_name: str) -> str:
        return file_name.rsplit(".", 1)[0]

    def _get_reader(self, file_extension: str) -> FileReader:
        return self.registry.get(file_extension)


class DocumentIngestionService:
    def __init__(
        self,
        uow: UnitOfWork,
        splitter: TextSplitter,
        embedder: TextEmbedder,
    ) -> None:
        self.uow = uow
        self.splitter = splitter
        self.embedder = embedder

    def add_document(
        self, text: str, workspace_id: int, title: str
    ) -> tuple[int, str, str]:
        with self.uow:
            workspace = self.uow.workspaces.get(workspace_id)
            if workspace is None:
                raise WorkspaceNotFound(workspace_id)

            document = Document(
                content=Content(text),
                workspace=workspace,
                title=title or None,
            )
            self.uow.documents.add(document)
            self.uow.commit()
            document_id = document.document_id  # type: ignore[attr-defined]

        self.ingest_document(document_id)
        return document_id, title, text

    @observe(name="ingest-document", as_type="chain")
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

    @observe(name="ingest-workspace", as_type="chain")
    def ingest_workspace(self, workspace_id: int) -> None:
        with self.uow:
            if self.uow.workspaces.get(workspace_id) is None:
                raise WorkspaceNotFound(workspace_id)

            documents, _ = self.uow.documents.list_unpreprocessed_by_workspace(
                workspace_id
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


class InputValidatorService:
    def __init__(self, input_sanitizer: InputSanitizer, pii_redactor: PIIRedactor):
        self.input_sanitizer = input_sanitizer
        self.pii_redactor = pii_redactor

    def validate(self, text: str) -> str:
        sanitized = self.input_sanitizer.sanitize(text)
        redacted = self.pii_redactor.redact(sanitized)
        return redacted


class OutputValidatorService:
    def __init__(self, output_validator: OutputValidator):
        self.output_validator = output_validator

    def validate(self, text: str) -> str:
        return self.output_validator.validate(text)


class RagService:
    def __init__(
        self,
        uow: UnitOfWork,
        embedder: TextEmbedder,
        llm: LLMChat,
        top_k: int,
        chat_prompt: PromptTemplate,
    ) -> None:
        self.uow = uow
        self.embedder = embedder
        self.llm = llm
        self.top_k = top_k
        self.chat_prompt = chat_prompt

    @observe(name="retrieve-context", as_type="retriever")
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

    @observe(name="rag-query", as_type="chain")
    def query(self, workspace_id: int, query_text: str) -> str:
        context = self._retrieve(workspace_id, query_text)
        prompt_text = self.chat_prompt.format(context=context, question=query_text)
        return self.llm.invoke(prompt_text)
