from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from app.db.session import DatabaseSessionFactory

FOLLOW_UP_OPTIONS = ("待跟进", "跟进中", "已发放测试账号", "已完成")
TEST_ACCOUNT_OPTIONS = ("待发放", "已发放")
DEFAULT_PAGE_SIZE = 10
MAX_PAGE_SIZE = 50

# 仅展示访客在「测试账号申请」弹窗中点击「提交申请」后的记录
_TRIAL_APPLY_FILTER = """(
    COALESCE(json_extract(p.interests_json, '$._session.trial_apply_submitted'), 0) = 1
    OR (
        COALESCE(json_extract(p.interests_json, '$._lead.wants_trial'), 0) = 1
        AND trim(coalesce(p.display_name, '')) != ''
        AND trim(coalesce(p.org_name, '')) != ''
        AND COALESCE(json_extract(p.interests_json, '$._session.trial_account_issued'), 0) = 1
    )
)"""
_NOT_DELETED_FILTER = "COALESCE(json_extract(p.interests_json, '$._admin.deleted'), 0) != 1"
_VISIBLE_CUSTOMER_WHERE = f"({_TRIAL_APPLY_FILTER}) AND {_NOT_DELETED_FILTER}"


@dataclass(frozen=True)
class AdminCustomerRow:
    session_id: str
    display_name: str
    org_name: str
    contact: str
    follow_up_status: str
    trial_account: str
    updated_at: str


class AdminCustomerRepository:
    def __init__(self, session_factory: DatabaseSessionFactory) -> None:
        self._session_factory = session_factory

    async def list_customers(
        self,
        *,
        query: str = "",
        page: int = 1,
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> Tuple[List[AdminCustomerRow], int]:
        q = (query or "").strip()
        page_num = max(1, int(page))
        size = max(1, min(int(page_size), MAX_PAGE_SIZE))
        offset = (page_num - 1) * size
        search_sql, search_params = _search_params(q)
        conn = await self._session_factory.connect()
        try:
            count_cur = await conn.execute(
                f"""
                SELECT COUNT(*)
                FROM chat_session_profiles p
                WHERE {_VISIBLE_CUSTOMER_WHERE}
                {search_sql}
                """,
                tuple(search_params),
            )
            count_row = await count_cur.fetchone()
            total = int(count_row[0] if count_row else 0)

            cur = await conn.execute(
                f"""
                SELECT p.session_id,
                       p.display_name,
                       p.org_name,
                       p.interests_json,
                       COALESCE(p.updated_at, '') AS updated_at
                FROM chat_session_profiles p
                WHERE {_VISIBLE_CUSTOMER_WHERE}
                {search_sql}
                ORDER BY COALESCE(p.updated_at, p.session_id) DESC
                LIMIT ? OFFSET ?
                """,
                tuple(search_params + [size, offset]),
            )
            rows = await cur.fetchall()
            leads = await self._load_leads_by_session(conn, [str(row["session_id"]) for row in rows])
            items = [_customer_row_from_db(row, leads.get(str(row["session_id"]), [])) for row in rows]
            return items, total
        finally:
            await conn.close()

    async def count_customers(self, *, query: str = "") -> int:
        q = (query or "").strip()
        search_sql, search_params = _search_params(q)
        conn = await self._session_factory.connect()
        try:
            cur = await conn.execute(
                f"""
                SELECT COUNT(*)
                FROM chat_session_profiles p
                WHERE {_VISIBLE_CUSTOMER_WHERE}
                {search_sql}
                """,
                tuple(search_params),
            )
            row = await cur.fetchone()
            return int(row[0] if row else 0)
        finally:
            await conn.close()

    async def count_trial_issued(self) -> int:
        conn = await self._session_factory.connect()
        try:
            cur = await conn.execute(
                f"""
                SELECT p.interests_json
                FROM chat_session_profiles p
                WHERE {_VISIBLE_CUSTOMER_WHERE}
                """
            )
            rows = await cur.fetchall()
            total = 0
            for row in rows:
                interests = _safe_json_obj(row["interests_json"])
                if _test_account_status(interests) == "已发放":
                    total += 1
            return total
        finally:
            await conn.close()

    async def get_customer(self, *, session_id: str) -> Optional[AdminCustomerRow]:
        sid = (session_id or "").strip()
        if not sid:
            return None
        conn = await self._session_factory.connect()
        try:
            cur = await conn.execute(
                f"""
                SELECT p.session_id,
                       p.display_name,
                       p.org_name,
                       p.interests_json,
                       COALESCE(p.updated_at, '') AS updated_at
                FROM chat_session_profiles p
                WHERE p.session_id = ?
                  AND {_VISIBLE_CUSTOMER_WHERE}
                LIMIT 1
                """,
                (sid,),
            )
            row = await cur.fetchone()
            if not row:
                return None
            leads = await self._load_leads_by_session(conn, [sid])
            return _customer_row_from_db(row, leads.get(sid, []))
        finally:
            await conn.close()

    async def update_follow_up(self, *, session_id: str, follow_up_status: str) -> Optional[AdminCustomerRow]:
        sid = (session_id or "").strip()
        if not sid:
            return None
        status = _normalize_follow_up(follow_up_status)
        conn = await self._session_factory.connect()
        try:
            cur = await conn.execute(
                f"""
                SELECT interests_json
                FROM chat_session_profiles p
                WHERE p.session_id = ?
                  AND {_VISIBLE_CUSTOMER_WHERE}
                LIMIT 1
                """,
                (sid,),
            )
            row = await cur.fetchone()
            if not row:
                return None
            interests = _safe_json_obj(row["interests_json"])
            admin_meta = dict(interests.get("_admin") or {})
            admin_meta["follow_up_status"] = status
            interests["_admin"] = admin_meta
            await conn.execute(
                "UPDATE chat_session_profiles SET interests_json=?, updated_at=CURRENT_TIMESTAMP WHERE session_id=?",
                (json.dumps(interests, ensure_ascii=False), sid),
            )
            await conn.commit()
        finally:
            await conn.close()
        return await self.get_customer(session_id=sid)

    async def update_test_account(self, *, session_id: str, test_account_status: str) -> Optional[AdminCustomerRow]:
        sid = (session_id or "").strip()
        if not sid:
            return None
        status = _normalize_test_account(test_account_status)
        conn = await self._session_factory.connect()
        try:
            cur = await conn.execute(
                f"""
                SELECT interests_json
                FROM chat_session_profiles p
                WHERE p.session_id = ?
                  AND {_VISIBLE_CUSTOMER_WHERE}
                LIMIT 1
                """,
                (sid,),
            )
            row = await cur.fetchone()
            if not row:
                return None
            interests = _safe_json_obj(row["interests_json"])
            admin_meta = dict(interests.get("_admin") or {})
            admin_meta["test_account_status"] = status
            interests["_admin"] = admin_meta
            await conn.execute(
                "UPDATE chat_session_profiles SET interests_json=?, updated_at=CURRENT_TIMESTAMP WHERE session_id=?",
                (json.dumps(interests, ensure_ascii=False), sid),
            )
            await conn.commit()
        finally:
            await conn.close()
        return await self.get_customer(session_id=sid)

    async def delete_customer(self, *, session_id: str) -> bool:
        sid = (session_id or "").strip()
        if not sid:
            return False
        conn = await self._session_factory.connect()
        try:
            cur = await conn.execute(
                f"""
                SELECT interests_json
                FROM chat_session_profiles p
                WHERE p.session_id = ?
                  AND {_VISIBLE_CUSTOMER_WHERE}
                LIMIT 1
                """,
                (sid,),
            )
            row = await cur.fetchone()
            if not row:
                return False
            interests = _safe_json_obj(row["interests_json"])
            admin_meta = dict(interests.get("_admin") or {})
            admin_meta["deleted"] = True
            interests["_admin"] = admin_meta
            await conn.execute(
                "UPDATE chat_session_profiles SET interests_json=?, updated_at=CURRENT_TIMESTAMP WHERE session_id=?",
                (json.dumps(interests, ensure_ascii=False), sid),
            )
            await conn.commit()
            return True
        finally:
            await conn.close()

    async def _load_leads_by_session(self, conn: Any, session_ids: List[str]) -> dict[str, List[tuple[str, str]]]:
        if not session_ids:
            return {}
        placeholders = ",".join("?" for _ in session_ids)
        cur = await conn.execute(
            f"SELECT session_id, contact_type, contact_value FROM lead_captures WHERE session_id IN ({placeholders})",
            tuple(session_ids),
        )
        rows = await cur.fetchall()
        out: dict[str, List[tuple[str, str]]] = {}
        for row in rows:
            sid = str(row["session_id"])
            out.setdefault(sid, []).append((str(row["contact_type"] or ""), str(row["contact_value"] or "")))
        return out


def _search_params(query: str) -> tuple[str, list[Any]]:
    q = (query or "").strip()
    if not q:
        return "", []
    like = f"%{q}%"
    return (
        """
        AND (
            COALESCE(p.display_name, '') LIKE ?
            OR COALESCE(p.org_name, '') LIKE ?
            OR EXISTS (
                SELECT 1 FROM lead_captures l
                WHERE l.session_id = p.session_id
                  AND l.contact_value LIKE ?
            )
        )
        """,
        [like, like, like],
    )


def _customer_row_from_db(row: Any, leads: List[tuple[str, str]]) -> AdminCustomerRow:
    interests = _safe_json_obj(row["interests_json"])
    contact = _format_contact(leads, interests)
    return AdminCustomerRow(
        session_id=str(row["session_id"] or ""),
        display_name=str(row["display_name"] or "").strip(),
        org_name=str(row["org_name"] or "").strip(),
        contact=contact,
        follow_up_status=_follow_up_status(interests),
        trial_account=_test_account_status(interests),
        updated_at=str(row["updated_at"] or ""),
    )


def _format_contact(leads: List[tuple[str, str]], interests: dict[str, Any]) -> str:
    parts: List[str] = []
    for contact_type, contact_value in leads:
        value = (contact_value or "").strip()
        if not value:
            continue
        label = _contact_type_label(contact_type)
        parts.append(f"{label}{value}" if label else value)
    if parts:
        return " / ".join(parts)
    lead = interests.get("_lead") if isinstance(interests.get("_lead"), dict) else {}
    value = str(lead.get("contact_value") or "").strip()
    if value:
        return f"{_contact_type_label(str(lead.get('contact_type') or ''))}{value}"
    return ""


def _contact_type_label(contact_type: str) -> str:
    ct = (contact_type or "").strip().lower()
    if ct in {"wechat", "wx", "weixin"}:
        return "微信："
    if ct in {"phone", "mobile", "tel"}:
        return ""
    return ""


def _follow_up_status(interests: dict[str, Any]) -> str:
    admin_meta = interests.get("_admin") if isinstance(interests.get("_admin"), dict) else {}
    stored = str(admin_meta.get("follow_up_status") or "").strip()
    if stored in FOLLOW_UP_OPTIONS:
        return stored
    return "待跟进"


def _test_account_status(interests: dict[str, Any]) -> str:
    admin_meta = interests.get("_admin") if isinstance(interests.get("_admin"), dict) else {}
    stored = str(admin_meta.get("test_account_status") or "").strip()
    if stored in TEST_ACCOUNT_OPTIONS:
        return stored
    return "待发放"


def _normalize_follow_up(value: str) -> str:
    status = (value or "").strip()
    if status in FOLLOW_UP_OPTIONS:
        return status
    return "待跟进"


def _normalize_test_account(value: str) -> str:
    status = (value or "").strip()
    if status in TEST_ACCOUNT_OPTIONS:
        return status
    return "待发放"


def _safe_json_obj(raw: Any) -> dict[str, Any]:
    try:
        if raw is None:
            return {}
        if isinstance(raw, dict):
            return raw
        text = str(raw or "").strip()
        if not text:
            return {}
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}
