# HippoDocs

HippoDocs is a self-hosted API for asking questions about your documents. It stores document chunks in PostgreSQL with pgvector, retrieves relevant chunks for each question, and sends them to an OpenAI-compatible or Ollama language model.

The API uses FastAPI and supports PDF and Markdown uploads, configurable embedding and language model providers, optional Langfuse tracing, and input and output checks for common security and privacy risks.

## Requirements

- Python 3.13 or later
- [uv](https://docs.astral.sh/uv/)
- Docker Compose
- An embedding model and a chat model from Ollama or an OpenAI-compatible service

## Run locally

Install dependencies and create a local environment file:

```bash
uv sync
cp .env.example .env
```

Edit `.env` with your PostgreSQL URL and model provider settings. Then start PostgreSQL with pgvector, apply the database migrations, and run the API:

```bash
docker compose up -d db
uv run alembic upgrade head
uv run python -m app.main
```

The API listens at `http://localhost:8000`. Interactive API documentation is available at `http://localhost:8000/docs`.

## API

All application routes use the `/v1` prefix.

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/workspaces/` | Create a workspace |
| `GET` | `/v1/workspaces/` | List workspaces |
| `GET` | `/v1/workspaces/{workspace_id}` | Get a workspace |
| `DELETE` | `/v1/workspaces/{workspace_id}` | Delete a workspace |
| `POST` | `/v1/workspaces/{workspace_id}/documents` | Upload a PDF or Markdown document |
| `GET` | `/v1/workspaces/{workspace_id}/documents` | List workspace documents |
| `DELETE` | `/v1/workspaces/{workspace_id}/documents/{document_id}` | Delete a document |
| `POST` | `/v1/workspaces/{workspace_id}/ask/` | Ask a question about workspace documents |
| `GET` | `/health` | Check API health |

## Configuration

The application reads settings from environment variables or `.env`. The example file lists provider and Langfuse settings. Core settings include:

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | empty | PostgreSQL connection URL |
| `EMBEDDING_PROVIDER` | `ollama` | Embedding provider, `ollama` or `openai` |
| `EMBEDDING_BASE_URL` | `http://localhost:11434` | Embedding service URL |
| `EMBEDDING_MODEL` | empty | Embedding model name |
| `LLM_PROVIDER` | `openai` | Chat provider, `ollama` or `openai` |
| `LLM_BASE_URL` | `http://localhost:3001/v1` | Chat service URL |
| `LLM_MODEL` | empty | Chat model name |
| `OPENAI_API_KEY` | `no-key` | API key for OpenAI-compatible services |
| `MAX_FILESIZE` | `10485760` | Maximum upload size in bytes |
| `CHUNK_SIZE` | `512` | Document chunk size |
| `CHUNK_OVERLAP` | `50` | Overlap between document chunks |
| `TOP_K` | `25` | Number of chunks retrieved per question |

Both embedding and chat providers accept optional fallback settings. Langfuse tracing starts when its public key, secret key, and base URL are configured.

## Project status

HippoDocs is under development. The Compose configuration starts the database; run the API separately with the command above.

## License

See [LICENCE](LICENCE).
