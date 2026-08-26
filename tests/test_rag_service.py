import pytest

from app.domain.models import Chunk, Content, Document, Embedding
from app.services.rag_service import RagService, UnknownWorkspaceError
from app.services.uow import UnitOfWork
from tests.conftest import FakeEmbedder, FakeLLMChat, seed_workspace


@pytest.fixture
def llm() -> FakeLLMChat:
    return FakeLLMChat()


@pytest.fixture
def rag_service(uow: UnitOfWork, llm: FakeLLMChat) -> RagService:
    return RagService(
        uow=uow,
        embedder=FakeEmbedder(),
        llm=llm,
        top_k=2,
    )


def seed_workspace_with_chunks(uow: UnitOfWork, n_chunks: int = 2) -> int:
    workspace_id = seed_workspace(uow)
    with uow:
        workspace = uow.workspaces.get(workspace_id)
        assert workspace is not None

        document = Document(content=Content("doc body"), workspace=workspace)
        uow.documents.add(document)
        uow.chunks.save_all(
            [
                Chunk(
                    document_id=document.document_id,  # type: ignore[attr-defined]
                    content=Content(f"chunk{i}"),
                ).add_embedding(Embedding(vector=(0.1, 0.1), model_id="test"))
                for i in range(n_chunks)
            ]
        )
        uow.commit()
    return workspace_id


def test_query_unknown_workspace_raises_error(rag_service: RagService):
    with pytest.raises(UnknownWorkspaceError):
        rag_service.query(workspace_id=999, query_text="question")


def test_query_returns_llm_answer(uow: UnitOfWork, rag_service: RagService) -> None:
    workspace_id = seed_workspace_with_chunks(uow)

    assert rag_service.query(workspace_id, "what?") == "Answer"


def test_query_builds_prompt_from_retrieved_context_and_question(
    uow: UnitOfWork, rag_service: RagService, llm: FakeLLMChat
) -> None:
    workspace_id = seed_workspace_with_chunks(uow)
    rag_service.query(workspace_id, "what?")
    prompt = llm.prompts[0]

    assert "chunk0\nchunk1" in prompt
    assert "Question: what?" in prompt


def test_query_with_no_chunks_still_answers_with_empty_context(
    uow: UnitOfWork, rag_service: RagService, llm: FakeLLMChat
) -> None:
    workspace_id = seed_workspace(uow)
    rag_service.query(workspace_id, "what?")

    assert "Question: what?" in llm.prompts[0]
