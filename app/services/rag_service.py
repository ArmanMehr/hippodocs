from typing import BinaryIO

import structlog
from langfuse import observe

from app.adapters.file_reader import FileReader, FileReaderRegistry
from app.domain.models import (
    Content,
    Document,
    DocumentStatus,
    EmptyContentError,
    Workspace,
)
from app.exceptions import (
    DocumentNotFound,
    EmptyContent,
    ServerError,
    UnsupportedDocument,
    WorkspaceNotFound,
)
from app.services.ports import (
    InputValidator,
    LLMChat,
    OutputValidator,
    PromptTemplate,
    Retriever,
    TextEmbedder,
    TextSplitter,
)
from app.services.uow import UnitOfWork

logger = structlog.get_logger(__name__)


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
                raise DocumentNotFound()
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

    def new_document(self, workspace_id: int, title: str, text: str) -> int | None:
        with self.uow:
            workspace = self.uow.workspaces.get(workspace_id)
            if workspace is None:
                raise WorkspaceNotFound()

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
                raise WorkspaceNotFound()
            document = self.uow.documents.get(document_id)
            if document is None or document.workspace.workspace_id != workspace_id:  # type: ignore[attr-defined]
                raise DocumentNotFound()
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
        input_validator: InputValidator,
    ) -> None:
        self.uow = uow
        self.splitter = splitter
        self.embedder = embedder
        self.validator = input_validator

    def add_document(
        self, text: str, workspace_id: int, title: str
    ) -> tuple[int, str, str]:
        document_content, title_content = self._to_content(text, title)
        with self.uow:
            workspace = self._ensure_workspace(workspace_id)
            document = self._check_existing_document(
                workspace_id, document_content.sha256
            )

            if not document:
                content_valid = self.validator.validate(document_content)
                title_valid = self.validator.validate(title_content).value

                document = Document(
                    content=content_valid,
                    workspace=workspace,
                    title=title_valid,
                )
                self.uow.documents.add(document)
                self.uow.flush()

                saved_document_id = document.document_id  # type: ignore[attr-defined]
                self.uow.documents.add_content_hash(
                    workspace_id,
                    saved_document_id,
                    document_content.sha256,
                )
                self.uow.commit()
            else:
                logger.debug(f"Document already exists. The status: {document.status}")
                content_valid = document.content
                title_valid = document.title

            document_id = document.document_id  # type: ignore[attr-defined]
            _ = document.workspace.workspace_id  # type: ignore[attr-defined]

        self._ingest_document(document)
        return document_id, title_valid, content_valid.value

    def retry_ingestion_in_workspace(self, workspace_id: int) -> None:
        with self.uow:
            self._ensure_workspace(workspace_id)
            documents = self.uow.documents.list_undone_by_workspace(workspace_id)
            for document in documents:
                _ = document.workspace.workspace_id  # type: ignore[attr-defined]

        for document in documents:
            self._ingest_document(document)

    def _ingest_document(self, document: Document) -> None:
        while document.status != DocumentStatus.DONE:
            document = self._route_document(document)

    def _route_document(self, document: Document) -> Document:
        if document.status in (DocumentStatus.PENDING, DocumentStatus.CHUNKING_FAILED):
            return self._chunk(document)
        elif document.status in (DocumentStatus.CHUNKED, DocumentStatus.EMBED_FAILED):
            return self._embed(document)
        else:
            raise ServerError()

    def _chunk(self, document: Document) -> Document:
        workspace_id = document.workspace.workspace_id  # type: ignore[attr-defined]
        document_id = document.document_id  # type: ignore[attr-defined]

        try:
            chunks = self.splitter.split_text(document.text, document_id, workspace_id)
        except Exception:
            with self.uow:
                self.uow.documents.update_status(
                    document_id=document_id,
                    new_status=DocumentStatus.CHUNKING_FAILED,
                )
                self.uow.commit()
            document.status = DocumentStatus.CHUNKING_FAILED
            raise

        if not chunks:
            raise UnsupportedDocument()

        with self.uow:
            self.uow.chunks.save_all(chunks)
            self.uow.documents.update_status(document_id, DocumentStatus.CHUNKED)
            self.uow.commit()
        document.status = DocumentStatus.CHUNKED

        return document

    def _embed(self, document: Document) -> Document:
        document_id = document.document_id  # type: ignore[attr-defined]
        with self.uow:
            chunks = self.uow.chunks.get_by_document(document_id)

        try:
            embeddings = self.embedder.embed_texts([c.text for c in chunks])
            chunks_embedded = [
                c.add_embedding(emb) for c, emb in zip(chunks, embeddings, strict=True)
            ]
        except Exception:
            with self.uow:
                self.uow.documents.update_status(
                    document_id, DocumentStatus.EMBED_FAILED
                )
                self.uow.commit()
            raise

        with self.uow:
            self.uow.chunks.update_embeddings(chunks_embedded)
            self.uow.documents.update_status(document_id, DocumentStatus.DONE)
            self.uow.commit()
        document.status = DocumentStatus.DONE
        return document

    def _ensure_workspace(self, workspace_id: int) -> Workspace:
        workspace = self.uow.workspaces.get(workspace_id)
        if workspace is None:
            raise WorkspaceNotFound()
        return workspace

    def _check_existing_document(self, workspace_id: int, content_hash: str):
        return self.uow.documents.get_identical_document_in_workspace(
            workspace_id=workspace_id, content_hash=content_hash
        )

    def _to_content(self, text: str, title: str):
        try:
            document_content = Content(text)
        except EmptyContentError:
            raise EmptyContent("Document content cannot be empty or whitespace")

        try:
            title_content = Content(title)
        except EmptyContentError:
            raise EmptyContent(
                "Document title or file name cannot be empty or whitespace"
            )

        return document_content, title_content


class RagService:
    def __init__(
        self,
        uow: UnitOfWork,
        retriever: Retriever,
        llm: LLMChat,
        top_k: int,
        chat_prompt: PromptTemplate,
        input_validator: InputValidator,
        output_validator: OutputValidator,
    ) -> None:
        self.uow = uow
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.chat_prompt = chat_prompt
        self.input_validator = input_validator
        self.output_validator = output_validator

    def _ensure_workspace(self, workspace_id: int) -> None:
        with self.uow:
            workspace = self.uow.workspaces.get(workspace_id)
        if workspace is None:
            raise WorkspaceNotFound()

    @observe(
        name="rag-query",
        as_type="chain",
        capture_input=True,
        capture_output=True,
    )
    def query(self, workspace_id: int, query_text: str) -> str:
        self._ensure_workspace(workspace_id)
        query_content = self.input_validator.validate(Content(query_text))
        chunks_view = self.retriever.retrieve_in_workspace(
            workspace_id=workspace_id, query=query_content, top_k=self.top_k
        )
        context = "\n".join([c.text for c in chunks_view])
        prompt = self.chat_prompt.format(context=context, question=query_text)
        llm_output = self.llm.invoke(prompt)
        return self.output_validator.validate(llm_output).value
