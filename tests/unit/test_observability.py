import os
from collections.abc import Generator

import pytest
from pytest_mock import MockerFixture

from app import observability
from app.observability import LangfuseState


@pytest.fixture(autouse=True)
def reset_observability_state(mocker: MockerFixture) -> Generator[None]:
    mocker.patch.dict(os.environ, {"LANGFUSE_PUBLIC_KEY": ""})
    mocker.patch.dict(os.environ, {"LANGFUSE_SECRET_KEY": ""})
    observability._state = LangfuseState.UNINITIALIZED
    observability._client = None
    yield
    observability._state = LangfuseState.UNINITIALIZED
    observability._client = None


def test_initialize_langfuse_warns_only_once_when_not_configured(mocker: MockerFixture):
    warning_mock = mocker.patch.object(observability.logger, "warning")

    observability.initialize_langfuse()
    observability.initialize_langfuse()
    observability.initialize_langfuse()

    assert warning_mock.call_count == 1
    assert warning_mock.call_args[0][0] == "langfuse_not_configured"
    assert observability._state is LangfuseState.DISABLED


def test_get_langfuse_client_returns_none_when_disabled(mocker: MockerFixture):
    mocker.patch.object(observability.logger, "warning")

    assert observability.get_langfuse_client() is None
    assert observability.get_langfuse_client() is None
    assert observability._state is LangfuseState.DISABLED
