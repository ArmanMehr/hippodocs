# HippoDocs

A minimal RAG (Retrieval-Augmented Generation) API built with FastAPI, LangChain, and pgvector.

> **v0.0.1** — under development.

## To do list

- [x] Add standard error/exception handling.
- [ ] Add testing.
- [ ] Add monitoring and observability.
- [ ] Add a security layer for PII protection, input sanitization, and output validation.
- [ ] Support additional input file types.
- [x] Add workspace functionality so that each chat has its own sources.
- [ ] Add source citations and references to answers.
- [ ] Switch to asynchronus architecture.
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

| Method | Path | Description |
|--------|------|-------------|
| POST | `/workspaces/` | Create a workspace |
| GET | `/workspaces/` | List workspaces |
| GET | `/workspaces/{workspace_id}` | Get a workspace |
| DELETE | `/workspaces/{workspace_id}` | Delete a workspace |
| POST | `/workspaces/upload-document` | Upload a PDF to a workspace |
| POST | `/ask/` | Ask a question |
| GET | `/health` | Health check |

## Configuration

See `.env.example`. Supports Ollama and OpenAI-compatible providers for both embeddings and LLM, with optional fallback providers.

