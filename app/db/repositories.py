from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable, List, Optional

from app.data.splitter import TextChunk
from app.db.models import schema_statements
from app.db.session import DatabaseSessionFactory
from app.schemas.chat import ChatResponse


class DocumentRepository:
    def __init__(self, session_factory: DatabaseSessionFactory) -> None:
        self._session_factory = session_factory

    async def init_db(self) -> None:
        conn = await self._session_factory.connect()
        try:
            for stmt in schema_statements(dialect=self._session_factory.dialect):
                await conn.execute(stmt)
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


@dataclass(frozen=True)
class AdminVideoAssetRow:
    id: int
    scene_key: str
    scene_name: str
    title: str
    original_filename: str
    stored_filename: str
    file_path: str
    file_url: str
    mime_type: str
    file_size: int
    duration_seconds: int | None
    status: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class AdminSceneIntroRow:
    scene_key: str
    scene_name: str
    intro_text: str
    decision_intro_text: str
    user_intro_text: str
    created_at: str
    updated_at: str


class AdminVideoAssetRepository:
    def __init__(self, session_factory: DatabaseSessionFactory) -> None:
        self._session_factory = session_factory

    async def init_db(self) -> None:
        conn = await self._session_factory.connect()
        try:
            for stmt in schema_statements(dialect=self._session_factory.dialect)[9:11]:
                await conn.execute(stmt)
            await conn.commit()
        finally:
            await conn.close()

    async def insert_video(
        self,
        *,
        scene_key: str,
        scene_name: str,
        title: str,
        original_filename: str,
        stored_filename: str,
        file_path: str,
        file_url: str,
        mime_type: str,
        file_size: int,
        duration_seconds: int | None = None,
    ) -> AdminVideoAssetRow:
        conn = await self._session_factory.connect()
        try:
            asset_id = await conn.fetchval(
                "INSERT INTO admin_video_assets("
                "scene_key, scene_name, title, original_filename, stored_filename, file_path, file_url, "
                "mime_type, file_size, duration_seconds"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id",
                (
                    scene_key,
                    scene_name,
                    title,
                    original_filename,
                    stored_filename,
                    file_path,
                    file_url,
                    mime_type,
                    int(file_size),
                    duration_seconds,
                ),
            )
            await conn.commit()
            row = await self.get_video(asset_id=asset_id)
            if row is None:
                raise RuntimeError("inserted admin video asset not found")
            return row
        finally:
            await conn.close()

    async def get_video(self, *, asset_id: int) -> AdminVideoAssetRow | None:
        conn = await self._session_factory.connect()
        try:
            row = await conn.fetchone(
                "SELECT * FROM admin_video_assets WHERE id=? AND status='active'",
                (int(asset_id),),
            )
            return _admin_video_asset_from_row(row) if row else None
        finally:
            await conn.close()

    async def list_videos(self, *, scene_key: str | None = None) -> List[AdminVideoAssetRow]:
        conn = await self._session_factory.connect()
        try:
            if scene_key:
                rows = await conn.fetchall(
                    "SELECT * FROM admin_video_assets WHERE scene_key=? AND status='active' ORDER BY id DESC",
                    (scene_key,),
                )
            else:
                rows = await conn.fetchall(
                    "SELECT * FROM admin_video_assets WHERE status='active' ORDER BY id DESC",
                )
            return [_admin_video_asset_from_row(row) for row in rows]
        finally:
            await conn.close()

    async def delete_video(self, *, asset_id: int) -> AdminVideoAssetRow | None:
        row = await self.get_video(asset_id=asset_id)
        if row is None:
            return None
        conn = await self._session_factory.connect()
        try:
            await conn.execute(
                "UPDATE admin_video_assets SET status='deleted', updated_at=CURRENT_TIMESTAMP "
                "WHERE id=? AND status='active'",
                (int(asset_id),),
            )
            await conn.commit()
        finally:
            await conn.close()
        return row


class AdminSceneIntroRepository:
    def __init__(self, session_factory: DatabaseSessionFactory) -> None:
        self._session_factory = session_factory

    async def _ensure_columns(self, conn) -> None:
        if self._session_factory.is_postgres:
            await conn.execute(
                "ALTER TABLE admin_scene_intros ADD COLUMN IF NOT EXISTS decision_intro_text TEXT NOT NULL DEFAULT ''"
            )
            await conn.execute(
                "ALTER TABLE admin_scene_intros ADD COLUMN IF NOT EXISTS user_intro_text TEXT NOT NULL DEFAULT ''"
            )
            return
        try:
            await conn.execute(
                "ALTER TABLE admin_scene_intros ADD COLUMN decision_intro_text TEXT NOT NULL DEFAULT ''"
            )
        except Exception:
            pass
        try:
            await conn.execute(
                "ALTER TABLE admin_scene_intros ADD COLUMN user_intro_text TEXT NOT NULL DEFAULT ''"
            )
        except Exception:
            pass

    async def init_db(self) -> None:
        conn = await self._session_factory.connect()
        try:
            await conn.execute(schema_statements(dialect=self._session_factory.dialect)[11])
            await self._ensure_columns(conn)
            await conn.commit()
        finally:
            await conn.close()

    async def get_intro(self, *, scene_key: str) -> AdminSceneIntroRow | None:
        conn = await self._session_factory.connect()
        try:
            await self._ensure_columns(conn)
            row = await conn.fetchone(
                "SELECT * FROM admin_scene_intros WHERE scene_key=?",
                ((scene_key or "").strip(),),
            )
            return _admin_scene_intro_from_row(row) if row else None
        finally:
            await conn.close()

    async def list_intros(self) -> List[AdminSceneIntroRow]:
        conn = await self._session_factory.connect()
        try:
            await self._ensure_columns(conn)
            rows = await conn.fetchall("SELECT * FROM admin_scene_intros ORDER BY scene_key ASC")
            return [_admin_scene_intro_from_row(row) for row in rows]
        finally:
            await conn.close()

    async def upsert_intro(
        self,
        *,
        scene_key: str,
        scene_name: str,
        intro_text: str,
        decision_intro_text: str,
        user_intro_text: str,
    ) -> AdminSceneIntroRow:
        key = (scene_key or "").strip()
        conn = await self._session_factory.connect()
        try:
            await self._ensure_columns(conn)
            if self._session_factory.is_postgres:
                await conn.execute(
                    "INSERT INTO admin_scene_intros("
                    "scene_key, scene_name, intro_text, decision_intro_text, user_intro_text"
                    ") VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT(scene_key) DO UPDATE SET "
                    "scene_name=excluded.scene_name, "
                    "intro_text=excluded.intro_text, "
                    "decision_intro_text=excluded.decision_intro_text, "
                    "user_intro_text=excluded.user_intro_text, "
                    "updated_at=CURRENT_TIMESTAMP",
                    (
                        key,
                        (scene_name or "").strip(),
                        (intro_text or "").strip(),
                        (decision_intro_text or "").strip(),
                        (user_intro_text or "").strip(),
                    ),
                )
            else:
                await conn.execute(
                    "INSERT INTO admin_scene_intros("
                    "scene_key, scene_name, intro_text, decision_intro_text, user_intro_text"
                    ") VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT(scene_key) DO UPDATE SET "
                    "scene_name=excluded.scene_name, "
                    "intro_text=excluded.intro_text, "
                    "decision_intro_text=excluded.decision_intro_text, "
                    "user_intro_text=excluded.user_intro_text, "
                    "updated_at=CURRENT_TIMESTAMP",
                    (
                        key,
                        (scene_name or "").strip(),
                        (intro_text or "").strip(),
                        (decision_intro_text or "").strip(),
                        (user_intro_text or "").strip(),
                    ),
                )
            await conn.commit()
        finally:
            await conn.close()
        row = await self.get_intro(scene_key=key)
        if row is None:
            raise RuntimeError("upserted admin scene intro not found")
        return row


def _admin_video_asset_from_row(row) -> AdminVideoAssetRow:
    return AdminVideoAssetRow(
        id=int(row["id"]),
        scene_key=str(row["scene_key"] or ""),
        scene_name=str(row["scene_name"] or ""),
        title=str(row["title"] or ""),
        original_filename=str(row["original_filename"] or ""),
        stored_filename=str(row["stored_filename"] or ""),
        file_path=str(row["file_path"] or ""),
        file_url=str(row["file_url"] or ""),
        mime_type=str(row["mime_type"] or ""),
        file_size=int(row["file_size"] or 0),
        duration_seconds=int(row["duration_seconds"]) if row["duration_seconds"] is not None else None,
        status=str(row["status"] or ""),
        created_at=str(row["created_at"] or ""),
        updated_at=str(row["updated_at"] or ""),
    )


def _admin_scene_intro_from_row(row) -> AdminSceneIntroRow:
    return AdminSceneIntroRow(
        scene_key=str(row["scene_key"] or ""),
        scene_name=str(row["scene_name"] or ""),
        intro_text=str(row["intro_text"] or ""),
        decision_intro_text=str(row["decision_intro_text"] or ""),
        user_intro_text=str(row["user_intro_text"] or ""),
        created_at=str(row["created_at"] or ""),
        updated_at=str(row["updated_at"] or ""),
    )


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
            if self._session_factory.is_postgres:
                inserted = await conn.fetchval(
                    "INSERT INTO lead_captures(session_id, contact_type, contact_value, visitor_type) "
                    "VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(session_id, contact_type, contact_value) DO NOTHING "
                    "RETURNING id",
                    (sid, ct, cv, visitor_type),
                )
                await conn.commit()
                return inserted is not None
            await conn.execute(
                "INSERT OR IGNORE INTO lead_captures(session_id, contact_type, contact_value, visitor_type) "
                "VALUES (?, ?, ?, ?)",
                (sid, ct, cv, visitor_type),
            )
            await conn.commit()
            row = await conn.fetchone("SELECT changes()")
            n = int(row[0]) if row and row[0] is not None else 0
            return n > 0
        finally:
            await conn.close()

    async def has_lead_for_session(self, *, session_id: str) -> bool:
        sid = (session_id or "").strip()
        if not sid:
            return False
        conn = await self._session_factory.connect()
        try:
            row = await conn.fetchone(
                "SELECT 1 FROM lead_captures WHERE session_id=? LIMIT 1",
                (sid,),
            )
            return row is not None
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
            if self._session_factory.is_postgres:
                await conn.execute(
                    "INSERT INTO chat_sessions(session_id, chat_mode, advisor_role) VALUES (?, ?, ?) "
                    "ON CONFLICT(session_id) DO NOTHING",
                    (sid, (chat_mode or "visitor_sales"), (advisor_role or "sales")),
                )
            else:
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

    async def reset_session(self, *, session_id: str, chat_mode: str = "visitor_sales", advisor_role: str = "sales") -> None:
        """清空该 session 的历史消息，并重置会话衍生状态。用于“强制新会话”语义兜底。"""
        sid = (session_id or "").strip()
        if not sid:
            return
        conn = await self._session_factory.connect()
        try:
            await conn.execute("DELETE FROM chat_messages WHERE session_id=?", (sid,))
            if self._session_factory.is_postgres:
                await conn.execute(
                    "INSERT INTO chat_sessions(session_id, chat_mode, advisor_role, visitor_type, created_at, updated_at) "
                    "VALUES (?, ?, ?, NULL, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP) "
                    "ON CONFLICT(session_id) DO UPDATE SET "
                    "chat_mode=excluded.chat_mode, advisor_role=excluded.advisor_role, "
                    "visitor_type=NULL, created_at=excluded.created_at, updated_at=excluded.updated_at",
                    (sid, (chat_mode or "visitor_sales"), (advisor_role or "sales")),
                )
            else:
                await conn.execute(
                    "INSERT OR REPLACE INTO chat_sessions(session_id, chat_mode, advisor_role, visitor_type, created_at, updated_at) "
                    "VALUES (?, ?, ?, NULL, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
                    (sid, (chat_mode or "visitor_sales"), (advisor_role or "sales")),
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
            rows = await conn.fetchall(
                "SELECT role, content, created_at FROM chat_messages "
                "WHERE session_id=? ORDER BY id DESC LIMIT ?",
                (sid, int(limit)),
            )
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
            if self._session_factory.is_postgres:
                deleted = int(
                    await conn.fetchval(
                        "WITH deleted AS ("
                        "DELETE FROM chat_messages "
                        "WHERE created_at < CURRENT_TIMESTAMP - (? || ' days')::interval "
                        "RETURNING 1"
                        ") SELECT COUNT(*) FROM deleted",
                        (str(days),),
                    )
                    or 0
                )
            else:
                await conn.execute(
                    "DELETE FROM chat_messages WHERE created_at < datetime('now', ?)",
                    (f"-{days} days",),
                )
                await conn.commit()
                row = await conn.fetchone("SELECT changes()")
                deleted = int(row[0]) if row and row[0] is not None else 0
            # 清掉已无消息且也过期的会话（updated_at 作为近似）
            if self._session_factory.is_postgres:
                await conn.execute(
                    "DELETE FROM chat_sessions "
                    "WHERE updated_at < CURRENT_TIMESTAMP - (? || ' days')::interval "
                    "AND session_id NOT IN (SELECT DISTINCT session_id FROM chat_messages)",
                    (str(days),),
                )
            else:
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
