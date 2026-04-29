from __future__ import annotations

import hashlib
import json
from typing import Iterable

from app.data.splitter import TextChunk
from app.db.models import CHUNKS_DDL, DOCUMENTS_DDL, QA_LOGS_DDL
from app.db.session import DatabaseSessionFactory
from app.schemas.chat import ChatResponse


class DocumentRepository:
    def __init__(self, session_factory: DatabaseSessionFactory) -> None:
        self._session_factory = session_factory

    async def init_db(self) -> None:
        conn = await self._session_factory.connect()
        try:
            await conn.execute(DOCUMENTS_DDL)
            await conn.execute(CHUNKS_DDL)
            await conn.execute(QA_LOGS_DDL)
            await conn.commit()
        finally:
            await conn.close()

    async def replace_documents(self, chunks: Iterable[TextChunk]) -> None:
        chunk_list = list(chunks)
        documents = {}
        conn = await self._session_factory.connect()
        try:
            await conn.execute("DELETE FROM chunks")
            await conn.execute("DELETE FROM documents")
            for chunk in chunk_list:
                documents.setdefault(
                    chunk.doc_id,
                    {
                        "doc_id": chunk.doc_id,
                        "title": chunk.title,
                        "url": chunk.url,
                        "content_hash": hashlib.sha256(chunk.text.encode("utf-8")).hexdigest(),
                    },
                )
                await conn.execute(
                    "INSERT INTO chunks(chunk_id, doc_id, chunk_order, snippet) VALUES (?, ?, ?, ?)",
                    (chunk.chunk_id, chunk.doc_id, chunk.order, chunk.text[:500]),
                )
            for doc in documents.values():
                await conn.execute(
                    "INSERT INTO documents(doc_id, title, url, content_hash) VALUES (?, ?, ?, ?)",
                    (doc["doc_id"], doc["title"], doc["url"], doc["content_hash"]),
                )
            await conn.commit()
        finally:
            await conn.close()


class QALogRepository:
    def __init__(self, session_factory: DatabaseSessionFactory) -> None:
        self._session_factory = session_factory

    async def log_chat(self, *, question: str, response: ChatResponse) -> None:
        conn = await self._session_factory.connect()
        try:
            await conn.execute(
                "INSERT INTO qa_logs(question, answer, sources, fallback_used) VALUES (?, ?, ?, ?)",
                (
                    question,
                    response.answer,
                    json.dumps([source.model_dump() for source in response.sources], ensure_ascii=False),
                    1 if response.fallback_used else 0,
                ),
            )
            await conn.commit()
        finally:
            await conn.close()

