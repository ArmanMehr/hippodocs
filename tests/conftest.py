from collections.abc import Generator

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm.session import Session, sessionmaker

from app.adapters.orm import clear_mappers, metadata, start_mappers


@pytest.fixture
def in_memory_db() -> Engine:
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        conn.execute(text("PRAGMA foreign_keys = ON"))
    metadata.create_all(engine)
    return engine


@pytest.fixture
def session_factory(in_memory_db: Engine) -> Generator[sessionmaker[Session]]:
    start_mappers()
    yield sessionmaker(bind=in_memory_db)
    clear_mappers()


@pytest.fixture
def session(session_factory: sessionmaker[Session]) -> Session:
    return session_factory()


class FakeSplitter:
    pass


class FakeEmbeddingModel:
    pass
