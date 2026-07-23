# simple-rag

> **v0.0.1** — under development.

A minimal RAG (Retrieval-Augmented Generation) API built with FastAPI, LangChain, and pgvector.

## Quick start

```bash
cp .env.example .env    # edit to match your providers
docker compose up -d    # starts PostgreSQL with pgvector
uv run alembic upgrade head
uv run python -m app.main
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/documents/` | Ingest text |
| POST | `/documents/upload` | Ingest a PDF |
| GET | `/documents/` | List documents |
| GET | `/documents/{id}` | Get a document |
| DELETE | `/documents/{id}` | Delete a document |
| POST | `/ask/` | Ask a question |
| GET | `/health` | Health check |

## Configuration

See `.env.example`. Supports Ollama and OpenAI-compatible providers for both embeddings and LLM, with optional fallback providers.
