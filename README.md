# HippoDocs

HippoDocs is a self-hosted retrieval-augmented generation (RAG) API. It stores documents in PostgreSQL with pgvector, retrieves relevant chunks, and uses an OpenAI-compatible or Ollama model to answer questions.

The API is built with FastAPI, SQLAlchemy, LangChain, and structlog. It supports PDF and Markdown ingestion, optional Langfuse tracing, request IDs, rate limiting, PII redaction, prompt-injection checks, and output validation.

## Current status

- Version: `0.0.1`
- Python: `3.13+`
- The application is under development.
- Docker Compose currently provides the PostgreSQL/pgvector database. The API itself runs locally.

## Quick start

Install [uv](https://docs.astral.sh/uv/) and make sure Docker is available, then install the project dependencies:

```bash
uv sync
```

Create a `.env` file in the project root. At minimum, configure the database and the models you want to use:

```dotenv
DATABASE_URL=postgresql+psycopg2://username:password123@localhost:5432/vector_db

EMBEDDING_PROVIDER=ollama
EMBEDDING_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=<embedding-model>

LLM_PROVIDER=openai
LLM_BASE_URL=http://localhost:3001/v1
LLM_MODEL=<chat-model>
OPENAI_API_KEY=no-key
```

Start PostgreSQL with pgvector and apply the migrations:

```bash
docker compose up -d db
uv run alembic upgrade head
```

Start the API:

```bash
uv run python -m app.main
```

The development server listens on `http://localhost:8000`. OpenAPI documentation is available at `/docs`.

## Configuration

Settings are read from environment variables or `.env`. Important settings include:

| Variable | Default | Purpose |
|---|---|---|
| `ENV` | `dev` | Runtime environment. Use `prd` for JSON logs. |
| `LOG_LEVEL` | `DEBUG` | Root log level. |
| `DATABASE_URL` | empty | SQLAlchemy PostgreSQL connection URL. |
| `EMBEDDING_PROVIDER` | `ollama` | Primary embedding provider. |
| `EMBEDDING_BASE_URL` | `http://localhost:11434` | Primary embedding endpoint. |
| `EMBEDDING_MODEL` | empty | Primary embedding model. |
| `LLM_PROVIDER` | `openai` | Primary chat provider. |
| `LLM_BASE_URL` | `http://localhost:3001/v1` | OpenAI-compatible chat endpoint. |
| `LLM_MODEL` | empty | Chat model. |
| `OPENAI_API_KEY` | `no-key` | API key for OpenAI-compatible services. |
| `MAX_FILESIZE` | `10485760` | Maximum upload size in bytes. |
| `CHUNK_SIZE` | `512` | Document chunk size. |
| `CHUNK_OVERLAP` | `50` | Chunk overlap. |
| `TOP_K` | `25` | Number of chunks retrieved for a question. |

Both embedding and LLM providers support optional fallback settings: `EMBEDDING_FALLBACK_PROVIDER`, `EMBEDDING_FALLBACK_BASE_URL`, `EMBEDDING_FALLBACK_MODEL`, `LLM_FALLBACK_PROVIDER`, `LLM_FALLBACK_BASE_URL`, and `LLM_FALLBACK_MODEL`.

Langfuse tracing is disabled unless `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and either `LANGFUSE_BASE_URL` or `LANGFUSE_HOST` are configured.

## API

All application endpoints are under `/v1`.

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/workspaces/` | Create a workspace |
| `GET` | `/v1/workspaces/` | List workspaces |
| `GET` | `/v1/workspaces/{workspace_id}` | Get a workspace |
| `DELETE` | `/v1/workspaces/{workspace_id}` | Delete a workspace |
| `POST` | `/v1/workspaces/{workspace_id}/documents` | Upload a PDF or Markdown document |
| `GET` | `/v1/workspaces/{workspace_id}/documents` | List workspace documents |
| `DELETE` | `/v1/workspaces/{workspace_id}/documents/{document_id}` | Delete a document |
| `POST` | `/v1/workspaces/{workspace_id}/ask/` | Ask a question using workspace documents |
| `GET` | `/health` | Health check |

Requests may include an `X-Request-ID` header. If it is missing or invalid, the API generates one and returns it in the response.

### Error responses

Application exceptions use a consistent response shape:

```json
{
  "detail": "Workspace not found",
  "error_code": "workspace_not_found",
  "timestamp": "2026-09-28T10:00:00+00:00"
}
```

The exception hierarchy includes workspace and document lookup errors, file validation errors, rate-limit errors, database-unavailable errors, and upstream embedding or LLM errors.

## Security and observability

- Input is checked for common prompt-injection patterns.
- PII is redacted during document and question validation.
- Model output is checked for secrets and selected PII before it is returned.
- Requests use structured logs with request IDs, status codes, paths, and duration.
- Provider rate limits and upstream failures are translated into application exceptions.
- Langfuse tracing is available for API calls, ingestion, retrieval, embedding, LLM, and security operations.

## Project layout

```text
app/
├── api/v1/       # Versioned FastAPI routes
├── adapters/     # Database, repository, model, file, and security adapters
├── domain/       # Domain models and invariants
├── services/     # Use cases and service ports
├── migrations/   # Alembic migrations
├── exceptions.py # Application exception hierarchy and error payloads
├── observability.py # Langfuse integration and tracing helpers
└── main.py       # FastAPI application, middleware, logging, and lifespan
tests/
├── unit/
├── integration/
└── e2e/
```

## Planned work

- Add source citations to answers.
- Move request and provider work to an asynchronous architecture.
- Add user registration, authentication, and authorization.
- Add a Streamlit UI.
- Containerize the API alongside the database.
- Add continuous integration.
