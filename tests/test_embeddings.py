from langchain_core.documents import Document
from pytest_mock import MockerFixture
from sqlalchemy.orm import Session

from app.adapters.repository import SQLAlchemyDocumentRepository
from app.services.embedding import DocumentEmbeddingService, DocumentIngestionPipeline
from tests.conftest import (
    FakeDeduplicator,
    FakeEmbeddingModel,
    FakeRepository,
    FakeSplitter,
)


def test_embedding_service_filters_blank_documents_and_maps_vectors(
    fake_splitter: type[FakeSplitter], fake_embeddings: type[FakeEmbeddingModel]
):
    splitter = fake_splitter([])
    embeddings = fake_embeddings(vectors=[[1.0, 2.0, 3.0]])
    service = DocumentEmbeddingService(splitter=splitter, embeddings_model=embeddings)
    docs = [Document(page_content="  "), Document(page_content="Hello")]

    models = service.embed_documents(docs)

    assert embeddings.texts == ["Hello"]
    assert [m.content for m in models] == ["Hello"]
    assert models[0].embedding == [1.0, 2.0, 3.0]


def test_ingestion_pipeline_runs_split_dedup_embed_and_save(
    mocker: MockerFixture,
    fake_splitter: type[FakeSplitter],
    fake_embeddings: type[FakeEmbeddingModel],
    fake_deduplicator: type[FakeDeduplicator],
    fake_repository: type[FakeRepository],
):
    split_docs = [Document(page_content="Chunk 1"), Document(page_content="Chunk 2")]
    kept_docs = [Document(page_content="Chunk 2")]
    splitter = fake_splitter(split_docs)
    embeddings = fake_embeddings(vectors=[[9.0, 8.0, 7.0]])
    embed_service = DocumentEmbeddingService(
        splitter=splitter, embeddings_model=embeddings
    )
    repo = fake_repository(mocker.MagicMock(spec=Session))
    deduplicator = fake_deduplicator(
        kept_docs, repo_mock=mocker.MagicMock(spec=SQLAlchemyDocumentRepository)
    )
    pipeline = DocumentIngestionPipeline(
        embedding_service=embed_service,
        repository=repo,
        deduplicator=deduplicator,
    )

    models = pipeline.run([Document(page_content="input")])

    assert [doc.page_content for doc in splitter.seen] == ["input"]
    assert [doc.page_content for doc in deduplicator.seen] == [
        "Chunk 1",
        "Chunk 2",
    ]
    assert embeddings.texts == ["Chunk 2"]
    assert [model.content for model in models] == ["Chunk 2"]
    assert repo.saved == models
