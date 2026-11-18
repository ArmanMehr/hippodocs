# HippoDocs

HippoDocs is a self-hosted API for uploading documents and asking questions about them. It stores document chunks in PostgreSQL with pgvector and ParadeDB BM25 indexing, retrieves relevant chunks, and sends them to an Ollama or OpenAI-compatible language model. PDF and Markdown files are supported. Input and output checks help detect common security and privacy risks, and Langfuse tracing is optional.

## Requirements

- Python 3.13 or later
- [uv](https://docs.astral.sh/uv/)
- Docker Compose
- A chat model and embedding model served by Ollama or an OpenAI-compatible API

## Run locally

Install the project dependencies and create a local environment file:

```bash
uv sync
cp .env.example .env
```

Edit `.env` to set the database URL and the model provider, URL, and model names. The Compose file starts the PostgreSQL database with the required ParadeDB and pgvector support. Apply migrations and start the API:

```bash
docker compose up -d db
uv run alembic upgrade head
uv run python -m app.main
```

The API listens at `http://localhost:8000`. Interactive API documentation is available at `http://localhost:8000/docs`.

## API

Application routes use the `/v1` prefix.

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/v1/workspaces/` | Create a workspace |
| `GET` | `/v1/workspaces/` | List workspaces (supports `skip` and `limit`) |
| `GET` | `/v1/workspaces/{workspace_id}` | Get a workspace |
| `DELETE` | `/v1/workspaces/{workspace_id}` | Delete a workspace |
| `POST` | `/v1/workspaces/{workspace_id}/documents` | Upload and ingest a PDF or Markdown file |
| `GET` | `/v1/workspaces/{workspace_id}/documents` | List workspace documents (supports `skip` and `limit`) |
| `DELETE` | `/v1/workspaces/{workspace_id}/documents/{document_id}` | Delete a document |
| `POST` | `/v1/workspaces/{workspace_id}/ask/` | Ask a question about workspace documents |
| `GET` | `/v1/health/live` | Liveness probe |
| `GET` | `/v1/health/ready` | Readiness probe for the database and both model providers |

Workspace creation expects JSON such as `{"name": "Research"}`. Document upload uses multipart form data with a `file` field. Questions use JSON such as `{"question": "What does the report say?"}`. List endpoints return a total count alongside the current page. The default maximum upload size is 5 MiB.

## Configuration

Settings can be supplied as environment variables or in `.env`. The names below are the primary settings for local setup; defaults come from `app/configs.py`.

| Variable | Default | Description |
| --- | --- | --- |
| `DATABASE_URL` | `sqlite:///:memory:` | SQLAlchemy database URL. Use the PostgreSQL URL from Compose for persistent local data. |
| `EMBEDDING_PROVIDER` | `openai` | Embedding provider: `openai` or `ollama`. |
| `EMBEDDING_BASE_URL` | empty | OpenAI-compatible embedding API URL; for Ollama, defaults to `http://localhost:11434`. |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model name. |
| `EMBEDDING_MAX_RETRIES` | `5` | Retry limit for OpenAI-compatible embedding requests. |
| `EMBEDDING_MAX_BATCH_SIZE` | `256` | Maximum embedding batch size. |
| `DIMENSIONS` | `384` | Embedding vector dimensions. The selected model must support this dimension. |
| `LLM_PROVIDER` | `openai` | Chat provider: `openai` or `ollama`. |
| `LLM_BASE_URL` | empty | OpenAI-compatible chat API URL; for Ollama, defaults to `http://localhost:11434`. |
| `LLM_MODEL` | `gpt-6-nano` | Chat model name. |
| `LLM_MAX_RETRIES` | `5` | Retry limit for OpenAI-compatible chat requests. |
| `OPENAI_API_KEY` | `no-key` | API key for OpenAI-compatible services. |
| `MAX_FILESIZE` | `5242880` | Maximum upload size in bytes (5 MiB). |
| `CHUNK_SIZE` | `1500` | Document chunk size. |
| `CHUNK_OVERLAP` | `200` | Overlap between document chunks. |
| `TOP_K` | `30` | Number of results used for vector retrieval. |
| `TOP_K_HYBRID` | `10` | Number of results used for hybrid retrieval. |
| `LANGFUSE_PUBLIC_KEY` | empty | Langfuse public key. |
| `LANGFUSE_SECRET_KEY` | empty | Langfuse secret key. |
| `LANGFUSE_BASE_URL` | empty | Langfuse server URL; falls back to `LANGFUSE_HOST`. |
| `LANGFUSE_HOST` | empty | Alternative Langfuse server URL. |

Langfuse tracing is disabled unless both keys are set. `ENV`, `LOG_LEVEL`, and `LOG_JSON` configure the environment and logging; see `app/configs.py` for the full settings list.

## Development

Run the test suite with:

```bash
uv run pytest
```

Run lint and type checks with:

```bash
uv run ruff check .
uv run pyrefly check
```

## License

See [LICENCE](LICENCE).
