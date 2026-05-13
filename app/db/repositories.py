from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable, List, Optional

from app.data.splitter import TextChunk
from app.db.models import (
    CHAT_MESSAGES_DDL,
    CHAT_MESSAGES_SESSION_CREATED_INDEX,
    CHAT_SESSIONS_DDL,
    CHUNKS_DDL,
    DOCUMENTS_DDL,
    LEAD_CAPTURES_DDL,
    LEAD_CAPTURES_UNIQUE_INDEX,
    QA_LOGS_DDL,
)
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
            await conn.execute(LEAD_CAPTURES_DDL)
            await conn.execute(LEAD_CAPTURES_UNIQUE_INDEX)
            await conn.execute(CHAT_SESSIONS_DDL)
            await conn.execute(CHAT_MESSAGES_DDL)
            await conn.execute(CHAT_MESSAGES_SESSION_CREATED_INDEX)
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


class LeadCaptureRepository:
    """访客留资：最小 SQLite 记录（v1.0）。"""

    def __init__(self, session_factory: DatabaseSessionFactory) -> None:
        self._session_factory = session_factory

    async def try_insert_lead(
        self,
        *,
        session_id: str,
        contact_type: str,
        contact_value: str,
        visitor_type: str | None,
    ) -> bool:
        """若同一 session 同联系方式已存在则跳过；成功插入返回 True。"""
        sid = (session_id or "").strip()
        ct = (contact_type or "").strip()
        cv = (contact_value or "").strip()
        if not sid or not ct or not cv:
            return False
        conn = await self._session_factory.connect()
        try:
            await conn.execute(
                "INSERT OR IGNORE INTO lead_captures(session_id, contact_type, contact_value, visitor_type) "
                "VALUES (?, ?, ?, ?)",
                (sid, ct, cv, visitor_type),
            )
            await conn.commit()
            cur2 = await conn.execute("SELECT changes()")
            row = await cur2.fetchone()
            n = int(row[0]) if row and row[0] is not None else 0
            return n > 0
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


@dataclass(frozen=True)
class ChatMessageRow:
    role: str
    content: str
    created_at: str


class ChatSessionRepository:
    """会话与消息：服务端持久化（v1，窗口读取）。"""

    def __init__(self, session_factory: DatabaseSessionFactory) -> None:
        self._session_factory = session_factory

    async def ensure_session(
        self,
        *,
        session_id: str,
        chat_mode: str,
        advisor_role: str = "sales",
    ) -> None:
        sid = (session_id or "").strip()
        if not sid:
            return
        conn = await self._session_factory.connect()
        try:
            await conn.execute(
                "INSERT OR IGNORE INTO chat_sessions(session_id, chat_mode, advisor_role) VALUES (?, ?, ?)",
                (sid, (chat_mode or "visitor_sales"), (advisor_role or "sales")),
            )
            await conn.execute(
                "UPDATE chat_sessions SET updated_at=CURRENT_TIMESTAMP WHERE session_id=?",
                (sid,),
            )
            await conn.commit()
        finally:
            await conn.close()

    async def update_visitor_type(self, *, session_id: str, visitor_type: Optional[str]) -> None:
        sid = (session_id or "").strip()
        vt = (visitor_type or "").strip()
        if not sid or not vt:
            return
        conn = await self._session_factory.connect()
        try:
            await conn.execute(
                "UPDATE chat_sessions SET visitor_type=?, updated_at=CURRENT_TIMESTAMP WHERE session_id=?",
                (vt, sid),
            )
            await conn.commit()
        finally:
            await conn.close()

    async def append_message(self, *, session_id: str, role: str, content: str) -> None:
        sid = (session_id or "").strip()
        c = (content or "").strip()
        r = (role or "").strip()
        if not sid or not r or not c:
            return
        conn = await self._session_factory.connect()
        try:
            await conn.execute(
                "INSERT INTO chat_messages(session_id, role, content) VALUES (?, ?, ?)",
                (sid, r, c),
            )
            await conn.execute(
                "UPDATE chat_sessions SET updated_at=CURRENT_TIMESTAMP WHERE session_id=?",
                (sid,),
            )
            await conn.commit()
        finally:
            await conn.close()

    async def list_recent_messages(self, *, session_id: str, limit: int) -> List[ChatMessageRow]:
        sid = (session_id or "").strip()
        if not sid or limit <= 0:
            return []
        conn = await self._session_factory.connect()
        try:
            cur = await conn.execute(
                "SELECT role, content, created_at FROM chat_messages "
                "WHERE session_id=? ORDER BY id DESC LIMIT ?",
                (sid, int(limit)),
            )
            rows = await cur.fetchall()
            out = [
                ChatMessageRow(
                    role=str(r["role"] or ""),
                    content=str(r["content"] or ""),
                    created_at=str(r["created_at"] or ""),
                )
                for r in rows
            ]
            out.reverse()
            return out
        finally:
            await conn.close()

    async def prune_older_than_days(self, *, retention_days: int) -> int:
        """删除过期消息与无消息的过期会话；返回删除的消息条数（近似）。"""
        days = int(retention_days)
        if days <= 0:
            return 0
        conn = await self._session_factory.connect()
        try:
            cur = await conn.execute(
                "DELETE FROM chat_messages WHERE created_at < datetime('now', ?)",
                (f"-{days} days",),
            )
            await conn.commit()
            cur2 = await conn.execute("SELECT changes()")
            row = await cur2.fetchone()
            deleted = int(row[0]) if row and row[0] is not None else 0
            # 清掉已无消息且也过期的会话（updated_at 作为近似）
            await conn.execute(
                "DELETE FROM chat_sessions "
                "WHERE updated_at < datetime('now', ?) "
                "AND session_id NOT IN (SELECT DISTINCT session_id FROM chat_messages)",
                (f"-{days} days",),
            )
            await conn.commit()
            return deleted
        finally:
            await conn.close()

