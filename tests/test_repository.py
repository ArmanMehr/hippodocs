from langchain_core.documents import Document
from pytest_mock import MockerFixture
from sqlalchemy.orm import Session

from app.models import Document as DocumentModel
from app.services.embedding import ExactMatchDeduplicator
from app.services.repository import DocumentRepository


def test_get_existing_contents_returns_lowercased_matches(mocker: MockerFixture):
    session = mocker.MagicMock(spec=Session)
    result = mocker.MagicMock()
    result.scalars.return_value.all.return_value = ["Alpha", "beta"]
    session.execute.return_value = result

    repo = DocumentRepository(session)

    assert repo.get_existing_contents(["ALPHA", "beta", "missing"]) == {
        "alpha",
        "beta",
    }
    session.execute.assert_called_once()


def test_save_all_commits_documents(mocker: MockerFixture):
    session = mocker.MagicMock(spec=Session)
    repo = DocumentRepository(session)
    docs = [DocumentModel(content="hello", embedding=[1.0, 2.0, 3.0])]

    repo.save_all(docs)

    session.add_all.assert_called_once_with(docs)
    session.commit.assert_called_once()


def test_list_all_returns_docs_and_total(mocker: MockerFixture):
    session = mocker.MagicMock(spec=Session)
    repo = DocumentRepository(session)
    docs = [DocumentModel(content="hello", embedding=[1.0, 2.0, 3.0])]
    session.scalar.return_value = 1
    session.scalars.return_value.all.return_value = docs

    found, total = repo.list_all(0, 20)

    assert found == docs
    assert total == 1
    session.scalar.assert_called_once()
    session.scalars.assert_called_once()


def test_delete_returns_false_when_missing(mocker: MockerFixture):
    session = mocker.MagicMock(spec=Session)
    repo = DocumentRepository(session)
    session.get.return_value = None

    assert repo.delete(123) is False
    session.delete.assert_not_called()
    session.commit.assert_not_called()


def test_delete_removes_document_when_found(mocker: MockerFixture):
    session = mocker.MagicMock(spec=Session)
    repo = DocumentRepository(session)
    doc = DocumentModel(content="hello", embedding=[1.0, 2.0, 3.0])
    session.get.return_value = doc

    assert repo.delete(123) is True
    session.delete.assert_called_once_with(doc)
    session.commit.assert_called_once()


def test_deduplicator_filters_existing_contents_case_insensitively(
    mocker: MockerFixture,
):
    repo = mocker.MagicMock(spec=DocumentRepository)
    repo.get_existing_contents.return_value = {"alpha"}
    deduplicator = ExactMatchDeduplicator(repo)
    docs = [
        Document(page_content="Alpha"),
        Document(page_content="Beta"),
    ]

    kept = deduplicator.filter_new(docs)

    assert [doc.page_content for doc in kept] == ["Beta"]
    repo.get_existing_contents.assert_called_once_with(["Alpha", "Beta"])
