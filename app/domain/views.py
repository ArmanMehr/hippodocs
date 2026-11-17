from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievedChunkView:
    chunk_id: int
    text: str
    relevance: float | None = None
