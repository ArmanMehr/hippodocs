from pytest_mock import MockerFixture

from app.adapters.repository import SQLAlchemyDocumentRepository
from app.services.agent import RagService
from tests.conftest import FakeEmbeddingModel, FakeSession


def test_rag_service_retrieve_uses_embeddings_and_session(
    mocker: MockerFixture,
    fake_embeddings: type[FakeEmbeddingModel],
    fake_session: type[FakeSession],
):
    embeddings = fake_embeddings()
    session = fake_session(["Doc A", "Doc B"])
    llm = mocker.MagicMock()
    repo = SQLAlchemyDocumentRepository(session)
    service = RagService(
        repository=repo,
        embeddings_model=embeddings,
        llm=llm,
        top_k=2,
    )

    result = service.retrieve("where are the docs")

    assert result == "Doc A\n\nDoc B"
    assert embeddings.queries == ["where are the docs"]
    assert session.last_stmt is not None
