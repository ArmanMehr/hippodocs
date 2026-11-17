from collections import defaultdict
from collections.abc import Sequence

from langfuse import observe
from sqlalchemy import text

from app.domain.models import Content
from app.domain.views import RetrievedChunkView
from app.services.ports import TextEmbedder
from app.services.uow import SQLAlchemyUnitOfWork


class PgVectorRetriever:
    def __init__(self, uow: SQLAlchemyUnitOfWork):
        self.uow = uow

    @observe(
        name="dense-retrieval",
        as_type="retriever",
        capture_input=False,
        capture_output=True,
    )
    def retrieve_in_document(
        self, document_id: int, query_vector: Sequence[float], top_k: int
    ) -> list[RetrievedChunkView]:
        stmt = text(
            """
            SELECT chunk_id, content
            FROM chunks
            WHERE chunks.document_id = :document_id
            AND chunks.embedding_vector IS NOT NULL
            ORDER BY chunks.embedding_vector <=> (:query_vector)::vector
            LIMIT :top_k
            """
        )
        with self.uow:
            results = self.uow.session.execute(
                stmt,
                {
                    "document_id": document_id,
                    "query_vector": str(list(query_vector)),
                    "top_k": top_k,
                },
            )
            return [
                RetrievedChunkView(chunk_id=row.chunk_id, text=row.content)
                for row in results
            ]

    @observe(
        name="dense-retrieval",
        as_type="retriever",
        capture_input=False,
        capture_output=True,
    )
    def retrieve_in_workspace(
        self, workspace_id: int, query_vector: Sequence[float], top_k: int
    ) -> list[RetrievedChunkView]:
        stmt = text(
            """
            SELECT chunk_id, content
            FROM chunks
            WHERE chunks.workspace_id = :workspace_id
            AND chunks.embedding_vector IS NOT NULL
            ORDER BY chunks.embedding_vector <=> (:query_vector)::vector
            LIMIT :top_k
            """
        )
        with self.uow:
            results = self.uow.session.execute(
                stmt,
                {
                    "workspace_id": workspace_id,
                    "query_vector": str(list(query_vector)),
                    "top_k": top_k,
                },
            )
            return [
                RetrievedChunkView(chunk_id=row.chunk_id, text=row.content)
                for row in results
            ]


class BM25Retriever:
    def __init__(self, uow: SQLAlchemyUnitOfWork):
        self.uow = uow

    @observe(
        name="sparse-retrieval",
        as_type="retriever",
        capture_input=False,
        capture_output=True,
    )
    def retrieve_in_document(
        self, document_id: int, query: Content, top_k: int
    ) -> list[RetrievedChunkView]:
        stmt = text(
            """
        SELECT
            chunk_id,
            content,
            pdb.score(chunk_id) AS relevance
        FROM chunks
        WHERE content ||| :query_text
        AND document_id = :document_id
        ORDER BY relevance DESC, chunk_id ASC
        LIMIT :top_k;
        """
        )

        with self.uow:
            results = self.uow.session.execute(
                stmt,
                {
                    "document_id": document_id,
                    "query_text": query.value,
                    "top_k": top_k,
                },
            )
            return [
                RetrievedChunkView(
                    chunk_id=row.chunk_id,
                    text=row.content,
                    relevance=row.relevance,
                )
                for row in results
            ]

    @observe(
        name="sparse-retrieval",
        as_type="retriever",
        capture_input=False,
        capture_output=True,
    )
    def retrieve_in_workspace(
        self, workspace_id: int, query: Content, top_k: int
    ) -> list[RetrievedChunkView]:
        stmt = text(
            """
        SELECT
            chunk_id,
            content,
            pdb.score(chunk_id) AS relevance
        FROM chunks
        WHERE content ||| :query_text
        AND workspace_id = :workspace_id
        ORDER BY relevance DESC, chunk_id ASC
        LIMIT :top_k;
        """
        )

        with self.uow:
            results = self.uow.session.execute(
                stmt,
                {
                    "workspace_id": workspace_id,
                    "query_text": query.value,
                    "top_k": top_k,
                },
            )
            return [
                RetrievedChunkView(
                    chunk_id=row.chunk_id,
                    text=row.content,
                    relevance=row.relevance,
                )
                for row in results
            ]


class HybridRetriever:
    def __init__(
        self,
        embedder: TextEmbedder,
        dense_retriever: PgVectorRetriever,
        sparse_retriever: BM25Retriever,
        top_k_single: int,
        smoothing_factor: int,
    ) -> None:
        self.embedder = embedder
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.top_k_single = top_k_single
        self.smoothing_factor = smoothing_factor

    def _reciprocal_rank_fusion(
        self, rankings: list[list[RetrievedChunkView]]
    ) -> list[RetrievedChunkView]:
        scores: dict[int, float] = defaultdict(float)
        chunk_map: dict[int, RetrievedChunkView] = {}
        for ranking in rankings:
            for rank, retrieved_chunk in enumerate(ranking, start=1):
                scores[retrieved_chunk.chunk_id] += 1.0 / (self.smoothing_factor + rank)
                if retrieved_chunk.chunk_id not in chunk_map:
                    chunk_map[retrieved_chunk.chunk_id] = retrieved_chunk
        sorted_chunk_ids = [s[0] for s in sorted(scores.items(), key=lambda x: -x[1])]
        return [chunk_map[chunk_id] for chunk_id in sorted_chunk_ids]

    def _extract_query_vector(self, query: Content) -> list[float]:
        query_embedding = self.embedder.embed_query(query.value)
        return list(query_embedding.vector)

    def _fuse_results(
        self,
        sparse_chunks: list[RetrievedChunkView],
        dense_chunks: list[RetrievedChunkView],
        top_k: int,
    ) -> list[RetrievedChunkView]:
        hybrid_chunks = self._reciprocal_rank_fusion([sparse_chunks, dense_chunks])
        return hybrid_chunks[:top_k]

    def retrieve_in_workspace(
        self, workspace_id: int, query: Content, top_k: int
    ) -> list[RetrievedChunkView]:
        query_vector = self._extract_query_vector(query)
        sparse_chunks = self.sparse_retriever.retrieve_in_workspace(
            workspace_id=workspace_id,
            query=query,
            top_k=self.top_k_single,
        )
        dense_chunks = self.dense_retriever.retrieve_in_workspace(
            workspace_id=workspace_id,
            query_vector=query_vector,
            top_k=self.top_k_single,
        )
        return self._fuse_results(sparse_chunks, dense_chunks, top_k)

    def retrieve_in_document(
        self, document_id: int, query: Content, top_k: int
    ) -> list[RetrievedChunkView]:
        query_vector = self._extract_query_vector(query)
        sparse_chunks = self.sparse_retriever.retrieve_in_document(
            document_id=document_id,
            query=query,
            top_k=self.top_k_single,
        )
        dense_chunks = self.dense_retriever.retrieve_in_document(
            document_id=document_id,
            query_vector=query_vector,
            top_k=self.top_k_single,
        )
        return self._fuse_results(sparse_chunks, dense_chunks, top_k)
