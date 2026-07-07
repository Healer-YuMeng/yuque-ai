from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

from app.db.session import DatabaseSessionFactory


@dataclass(frozen=True)
class StoredChunk:
    chunk_id: str
    doc_id: str
    title: str
    url: str
    text: str
    order: int


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: StoredChunk
    score: float


def normalize_vector(values: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(float(v) * float(v) for v in values))
    if norm <= 0:
        return [float(v) for v in values]
    return [float(v) / norm for v in values]


def format_pgvector(values: Sequence[float]) -> str:
    normalized = normalize_vector(values)
    return "[" + ",".join(f"{v:.8f}" for v in normalized) + "]"


class VectorStore:
    """PostgreSQL + pgvector backed vector storage and similarity search."""

    def __init__(self, *, session_factory: DatabaseSessionFactory) -> None:
        self._session_factory = session_factory

    async def search(self, query_embedding: Sequence[float], top_k: int) -> List[RetrievedChunk]:
        if not self._session_factory.is_postgres or top_k <= 0:
            return []

        query_vector = format_pgvector(query_embedding)
        conn = await self._session_factory.connect()
        try:
            rows = await conn.fetchall(
                """
                SELECT
                    c.chunk_id,
                    c.doc_id,
                    d.title,
                    d.url,
                    COALESCE(c.chunk_text, c.snippet) AS chunk_text,
                    c.chunk_order,
                    (1 - (c.embedding <=> ?::vector)) AS score
                FROM chunks c
                JOIN documents d ON d.doc_id = c.doc_id
                WHERE c.embedding IS NOT NULL
                ORDER BY c.embedding <=> ?::vector
                LIMIT ?
                """,
                (query_vector, query_vector, top_k),
            )
        finally:
            await conn.close()

        results: List[RetrievedChunk] = []
        for row in rows:
            results.append(
                RetrievedChunk(
                    chunk=StoredChunk(
                        chunk_id=str(row["chunk_id"]),
                        doc_id=str(row["doc_id"]),
                        title=str(row["title"] or ""),
                        url=str(row["url"] or ""),
                        text=str(row["chunk_text"] or ""),
                        order=int(row["chunk_order"] or 0),
                    ),
                    score=float(row["score"] or 0.0),
                )
            )
        return results

    async def chunk_count(self) -> int:
        if not self._session_factory.is_postgres:
            return 0
        conn = await self._session_factory.connect()
        try:
            value = await conn.fetchval("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL")
            return int(value or 0)
        finally:
            await conn.close()
