from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional  # noqa: F401 — Any used in _ensure_catalog_column

from app.conversation.catalog_state_machine import CatalogDialogState, dump_catalog_state_json, parse_catalog_state_json
from app.db.session import DatabaseSessionFactory


@dataclass(frozen=True)
class ChatSessionProfile:
    session_id: str
    display_name: str
    visitor_type: str
    org_name: str
    interests: Dict[str, Any]
    focused_doc_ids: List[str]


class ChatSessionProfileRepository:
    def __init__(self, session_factory: DatabaseSessionFactory) -> None:
        self._session_factory = session_factory

    async def _ensure_catalog_column(self, conn: Any) -> None:
        try:
            await conn.execute(
                "ALTER TABLE chat_session_profiles ADD COLUMN catalog_state_json TEXT NOT NULL DEFAULT '{}'"
            )
        except Exception:
            pass

    async def get_catalog_state(self, *, session_id: str) -> CatalogDialogState:
        sid = (session_id or "").strip()
        if not sid:
            return CatalogDialogState()
        conn = await self._session_factory.connect()
        try:
            await self._ensure_catalog_column(conn)
            cur = await conn.execute(
                "SELECT catalog_state_json FROM chat_session_profiles WHERE session_id=? LIMIT 1",
                (sid,),
            )
            row = await cur.fetchone()
            if not row:
                return CatalogDialogState()
            return parse_catalog_state_json(row["catalog_state_json"])
        finally:
            await conn.close()

    async def save_catalog_state(self, *, session_id: str, state: CatalogDialogState) -> None:
        sid = (session_id or "").strip()
        if not sid:
            return
        conn = await self._session_factory.connect()
        try:
            await self._ensure_catalog_column(conn)
            await conn.execute(
                "INSERT OR IGNORE INTO chat_session_profiles(session_id) VALUES (?)",
                (sid,),
            )
            await conn.execute(
                "UPDATE chat_session_profiles SET catalog_state_json=?, updated_at=CURRENT_TIMESTAMP WHERE session_id=?",
                (dump_catalog_state_json(state), sid),
            )
            await conn.commit()
        finally:
            await conn.close()

    async def get_profile(self, *, session_id: str) -> Optional[ChatSessionProfile]:
        sid = (session_id or "").strip()
        if not sid:
            return None
        conn = await self._session_factory.connect()
        try:
            await self._ensure_catalog_column(conn)
            cur = await conn.execute(
                "SELECT session_id, display_name, visitor_type, org_name, interests_json, focused_doc_ids_json "
                "FROM chat_session_profiles WHERE session_id=? LIMIT 1",
                (sid,),
            )
            row = await cur.fetchone()
            if not row:
                return None
            interests = _safe_json_obj(row["interests_json"], default={})
            focused = _safe_json_list(row["focused_doc_ids_json"], default=[])
            return ChatSessionProfile(
                session_id=str(row["session_id"] or sid),
                display_name=str(row["display_name"] or ""),
                visitor_type=str(row["visitor_type"] or ""),
                org_name=str(row["org_name"] or ""),
                interests=interests,
                focused_doc_ids=[str(x) for x in focused if str(x).strip()],
            )
        finally:
            await conn.close()

    async def upsert_profile(
        self,
        *,
        session_id: str,
        display_name: Optional[str] = None,
        visitor_type: Optional[str] = None,
        org_name: Optional[str] = None,
        interests: Optional[Dict[str, Any]] = None,
        focused_doc_ids: Optional[List[str]] = None,
    ) -> None:
        sid = (session_id or "").strip()
        if not sid:
            return
        conn = await self._session_factory.connect()
        try:
            await self._ensure_catalog_column(conn)
            await conn.execute(
                "INSERT OR IGNORE INTO chat_session_profiles(session_id) VALUES (?)",
                (sid,),
            )
            fields: List[str] = []
            values: List[Any] = []
            if display_name is not None:
                fields.append("display_name=?")
                values.append((display_name or "").strip())
            if visitor_type is not None:
                fields.append("visitor_type=?")
                values.append((visitor_type or "").strip())
            if org_name is not None:
                fields.append("org_name=?")
                values.append((org_name or "").strip())
            if interests is not None:
                fields.append("interests_json=?")
                values.append(json.dumps(interests, ensure_ascii=False))
            if focused_doc_ids is not None:
                uniq = []
                seen: set[str] = set()
                for x in focused_doc_ids:
                    v = str(x or "").strip()
                    if not v or v in seen:
                        continue
                    seen.add(v)
                    uniq.append(v)
                fields.append("focused_doc_ids_json=?")
                values.append(json.dumps(uniq, ensure_ascii=False))
            if fields:
                sql = "UPDATE chat_session_profiles SET " + ", ".join(fields) + ", updated_at=CURRENT_TIMESTAMP WHERE session_id=?"
                values.append(sid)
                await conn.execute(sql, tuple(values))
            await conn.commit()
        finally:
            await conn.close()

    async def touch_focus_docs(self, *, session_id: str, doc_ids: List[str], max_keep: int = 8) -> None:
        sid = (session_id or "").strip()
        if not sid:
            return
        current = await self.get_profile(session_id=sid)
        prev = list(current.focused_doc_ids) if current else []
        seen: set[str] = set()
        merged: List[str] = []
        for x in (doc_ids or []) + prev:
            v = str(x or "").strip()
            if not v or v in seen:
                continue
            seen.add(v)
            merged.append(v)
            if len(merged) >= max(1, int(max_keep)):
                break
        await self.upsert_profile(session_id=sid, focused_doc_ids=merged)


def _safe_json_obj(raw: Any, *, default: Dict[str, Any]) -> Dict[str, Any]:
    try:
        if raw is None:
            return dict(default)
        if isinstance(raw, (dict, list)):
            return raw if isinstance(raw, dict) else dict(default)
        s = str(raw or "").strip()
        if not s:
            return dict(default)
        v = json.loads(s)
        return v if isinstance(v, dict) else dict(default)
    except Exception:
        return dict(default)


def _safe_json_list(raw: Any, *, default: List[Any]) -> List[Any]:
    try:
        if raw is None:
            return list(default)
        if isinstance(raw, list):
            return raw
        s = str(raw or "").strip()
        if not s:
            return list(default)
        v = json.loads(s)
        return v if isinstance(v, list) else list(default)
    except Exception:
        return list(default)

