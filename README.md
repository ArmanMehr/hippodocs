# HippoDocs

A minimal RAG (Retrieval-Augmented Generation) API built with FastAPI, LangChain, and pgvector.

> **v0.0.1** — under development.

## To do list

- [x] Add standard error/exception handling.
- [x] RESTful, versioned endpoint layout (`/v1`).
- [x] Pluggable file readers (PDF, Markdown) via a registry.
- [ ] Add more test and increase coverage.
- [ ] Add monitoring and observability.
- [ ] Add a security layer for PII protection, input sanitization, and output validation.
- [ ] Support additional input file types.
- [x] Add workspace functionality so that each chat has its own sources.
- [ ] Add source citations and references to answers.
- [ ] Switch to asynchronous architecture.
- [ ] Implement user management, register, login, and authentication.
- [ ] Add a Streamlit UI.
- [ ] Dockerize the application.
- [ ] Implement a CI pipeline.

## Quick start

```bash
cp .env.example .env    # edit to match your providers
docker compose up -d    # starts PostgreSQL with pgvector
uv run alembic upgrade head
uv run python -m app.main
```

## Endpoints

All endpoints are served under the `/v1` prefix.

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/workspaces/` | Create a workspace |
| GET | `/v1/workspaces/` | List workspaces |
| GET | `/v1/workspaces/{workspace_id}` | Get a workspace |
| DELETE | `/v1/workspaces/{workspace_id}` | Delete a workspace |
| POST | `/v1/workspaces/{workspace_id}/documents` | Upload a document (PDF or Markdown) to a workspace |
| GET | `/v1/workspaces/{workspace_id}/documents` | List documents in a workspace |
| DELETE | `/v1/workspaces/{workspace_id}/documents/{document_id}` | Delete a document from a workspace |
| POST | `/v1/workspaces/{workspace_id}/ask` | Ask a question in a workspace |
| GET | `/v1/health` | Health check |

### Error responses

Domain exceptions (`WorkspaceNotFound`, `FileTooLarge`, `UnsupportedFileType`, `RateLimitError`, etc.) are caught by a central handler and returned as:

```json
{
  "detail": "Workspace 42 not found",
  "timestamp": "2025-01-01T12:00:00+00:00",
  "error_code": "workspace_not_found"
}
```

## Configuration

See `.env.example`. Supports Ollama and OpenAI-compatible providers for both embeddings and LLM, with optional fallback providers.

## Project layout

```
app/
├── api/
│   └── v1/              # Versioned FastAPI routers
│       ├── __init__.py
│       ├── ask.py
│       ├── documents.py
│       └── workspaces.py
├── adapters/            # Infrastructure adapters (LLM, embedders, file readers, ...)
├── domain/              # Domain models
├── services/            # Use-case services and ports
├── exceptions.py        # Domain exception hierarchy
└── main.py              # FastAPI app, global exception handler, lifespan
```
