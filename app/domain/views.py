from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievedChunkView:
    chunk_id: int
    content: str
    relevance: float | None = None
