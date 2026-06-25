from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from app.db.session import DatabaseSessionFactory

FOLLOW_UP_OPTIONS = ("待跟进", "跟进中", "已发放测试账号", "已完成")
TEST_ACCOUNT_OPTIONS = ("待发放", "已发放")
DEFAULT_PAGE_SIZE = 10
MAX_PAGE_SIZE = 50


@dataclass(frozen=True)
class AdminCustomerRow:
    session_id: str
    display_name: str
    org_name: str
    role_category: str
    contact: str
    email: str
    follow_up_status: str
    trial_account: str
    updated_at: str


class AdminCustomerRepository:
    def __init__(self, session_factory: DatabaseSessionFactory) -> None:
        self._session_factory = session_factory

    def _updated_at_select_expr(self) -> str:
        if self._session_factory.is_postgres:
            return "COALESCE(p.updated_at, l.latest_lead_at)::text"
        return "CAST(COALESCE(p.updated_at, l.latest_lead_at) AS TEXT)"

    def _order_by_sql(self) -> str:
        if self._session_factory.is_postgres:
            return """
                ORDER BY COALESCE(p.updated_at, l.latest_lead_at) DESC NULLS LAST,
                         COALESCE(p.session_id, l.session_id) DESC
            """
        return """
            ORDER BY COALESCE(p.updated_at, l.latest_lead_at) DESC,
                     COALESCE(p.session_id, l.session_id) DESC
        """

    def _not_deleted_filter(self) -> str:
        if self._session_factory.is_postgres:
            return "LOWER(COALESCE(p.interests_json::jsonb #>> '{_admin,deleted}', 'false')) NOT IN ('1', 'true')"
        return "LOWER(COALESCE(CAST(json_extract(p.interests_json, '$._admin.deleted') AS TEXT), '0')) NOT IN ('1', 'true')"

    def _customer_base_from(self) -> str:
        return """
            FROM (
                SELECT session_id, MAX(created_at) AS latest_lead_at
                FROM lead_captures
                GROUP BY session_id
            ) l
            LEFT JOIN chat_session_profiles p ON p.session_id = l.session_id
        """

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
        conn = await self._session_factory.connect()
        try:
            all_items = await self._load_deduped_customer_rows(conn, query=q)
            total = len(all_items)
            return all_items[offset: offset + size], total
        finally:
            await conn.close()

    async def _load_deduped_customer_rows(self, conn: Any, *, query: str = "") -> List[AdminCustomerRow]:
        q = (query or "").strip()
        search_sql, search_params = _search_params(q, is_postgres=self._session_factory.is_postgres)
        not_deleted_filter = self._not_deleted_filter()
        customer_from = self._customer_base_from()
        updated_at_expr = self._updated_at_select_expr()
        order_by_sql = self._order_by_sql()
        rows = await conn.fetchall(
            f"""
            SELECT COALESCE(p.session_id, l.session_id) AS session_id,
                   p.display_name,
                   p.org_name,
                   p.visitor_type,
                   p.interests_json,
                   {updated_at_expr} AS updated_at
            {customer_from}
            WHERE {not_deleted_filter}
            {search_sql}
            {order_by_sql}
            """,
            tuple(search_params),
        )
        leads = await self._load_leads_by_session(conn, [str(row["session_id"]) for row in rows])
        row_pairs = [
            (row, leads.get(str(row["session_id"]), []))
            for row in rows
        ]
        return _dedupe_customer_rows(row_pairs)

    async def count_customers(self, *, query: str = "") -> int:
        conn = await self._session_factory.connect()
        try:
            return len(await self._load_deduped_customer_rows(conn, query=query))
        finally:
            await conn.close()

    async def count_trial_issued(self) -> int:
        conn = await self._session_factory.connect()
        try:
            rows = await self._load_deduped_customer_rows(conn)
            return sum(1 for row in rows if row.trial_account == "已发放")
        finally:
            await conn.close()

    async def get_customer(self, *, session_id: str) -> Optional[AdminCustomerRow]:
        sid = (session_id or "").strip()
        if not sid:
            return None
        updated_at_expr = self._updated_at_select_expr()
        conn = await self._session_factory.connect()
        try:
            row = await conn.fetchone(
                f"""
                SELECT COALESCE(p.session_id, l.session_id) AS session_id,
                       p.display_name,
                       p.org_name,
                       p.visitor_type,
                       p.interests_json,
                       {updated_at_expr} AS updated_at
                {self._customer_base_from()}
                WHERE l.session_id = ?
                  AND {self._not_deleted_filter()}
                LIMIT 1
                """,
                (sid,),
            )
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
            row = await conn.fetchone(
                f"""
                SELECT interests_json
                {self._customer_base_from()}
                WHERE l.session_id = ?
                  AND {self._not_deleted_filter()}
                LIMIT 1
                """,
                (sid,),
            )
            if not row:
                return None
            interests = _safe_json_obj(row["interests_json"])
            admin_meta = dict(interests.get("_admin") or {})
            admin_meta["follow_up_status"] = status
            interests["_admin"] = admin_meta
            await self._ensure_profile_row(conn, session_id=sid)
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
            row = await conn.fetchone(
                f"""
                SELECT interests_json
                {self._customer_base_from()}
                WHERE l.session_id = ?
                  AND {self._not_deleted_filter()}
                LIMIT 1
                """,
                (sid,),
            )
            if not row:
                return None
            interests = _safe_json_obj(row["interests_json"])
            admin_meta = dict(interests.get("_admin") or {})
            admin_meta["test_account_status"] = status
            interests["_admin"] = admin_meta
            await self._ensure_profile_row(conn, session_id=sid)
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
            row = await conn.fetchone(
                f"""
                SELECT interests_json
                {self._customer_base_from()}
                WHERE l.session_id = ?
                  AND {self._not_deleted_filter()}
                LIMIT 1
                """,
                (sid,),
            )
            if not row:
                return False
            await conn.execute("DELETE FROM lead_captures WHERE session_id=?", (sid,))
            await conn.execute("DELETE FROM chat_messages WHERE session_id=?", (sid,))
            await conn.execute("DELETE FROM chat_session_profiles WHERE session_id=?", (sid,))
            await conn.execute("DELETE FROM chat_sessions WHERE session_id=?", (sid,))
            await conn.commit()
            return True
        finally:
            await conn.close()

    async def _ensure_profile_row(self, conn: Any, *, session_id: str) -> None:
        sid = (session_id or "").strip()
        if not sid:
            return
        if self._session_factory.is_postgres:
            await conn.execute(
                "INSERT INTO chat_session_profiles(session_id) VALUES (?) "
                "ON CONFLICT(session_id) DO NOTHING",
                (sid,),
            )
        else:
            await conn.execute(
                "INSERT OR IGNORE INTO chat_session_profiles(session_id) VALUES (?)",
                (sid,),
            )

    async def _load_leads_by_session(self, conn: Any, session_ids: List[str]) -> dict[str, List[tuple[str, str, str]]]:
        if not session_ids:
            return {}
        placeholders = ",".join("?" for _ in session_ids)
        rows = await conn.fetchall(
            f"SELECT session_id, contact_type, contact_value, visitor_type FROM lead_captures WHERE session_id IN ({placeholders})",
            tuple(session_ids),
        )
        out: dict[str, List[tuple[str, str, str]]] = {}
        for row in rows:
            sid = str(row["session_id"])
            out.setdefault(sid, []).append((
                str(row["contact_type"] or ""),
                str(row["contact_value"] or ""),
                str(row["visitor_type"] or ""),
            ))
        return out


def _search_params(query: str, *, is_postgres: bool) -> tuple[str, list[Any]]:
    q = (query or "").strip()
    if not q:
        return "", []
    like = f"%{q}%"
    ilike_op = "ILIKE" if is_postgres else "LIKE"
    return (
        """
        AND (
            COALESCE(p.display_name, '') """ + ilike_op + """ ?
            OR COALESCE(p.org_name, '') """ + ilike_op + """ ?
            OR EXISTS (
                SELECT 1 FROM lead_captures lc
                WHERE lc.session_id = l.session_id
                  AND lc.contact_value """ + ilike_op + """ ?
            )
            OR COALESCE(
                """ + ("p.interests_json::jsonb #>> '{_lead,email}'" if is_postgres else "CAST(json_extract(p.interests_json, '$._lead.email') AS TEXT)") + """,
                ''
            ) """ + ilike_op + """ ?
        )
        """,
        [like, like, like, like],
    )


def _customer_row_from_db(row: Any, leads: List[tuple[str, str, str]]) -> AdminCustomerRow:
    interests = _safe_json_obj(row["interests_json"])
    contact = _format_contact(leads, interests)
    return AdminCustomerRow(
        session_id=str(row["session_id"] or ""),
        display_name=str(row["display_name"] or "").strip(),
        org_name=str(row["org_name"] or "").strip(),
        role_category=_role_category(row, leads, interests),
        contact=contact,
        email=_lead_email(interests),
        follow_up_status=_follow_up_status(interests),
        trial_account=_test_account_status(interests),
        updated_at=str(row["updated_at"] or ""),
    )


def _dedupe_customer_rows(row_pairs: List[tuple[Any, List[tuple[str, str, str]]]]) -> List[AdminCustomerRow]:
    merged: dict[str, AdminCustomerRow] = {}
    for row, leads in row_pairs:
        item = _customer_row_from_db(row, leads)
        key = _customer_identity_key(row, leads)
        if key not in merged:
            merged[key] = item
            continue
        merged[key] = _merge_customer_row(merged[key], item)
    return list(merged.values())


def _customer_identity_key(row: Any, leads: List[tuple[str, str, str]]) -> str:
    for contact_type, contact_value, _visitor_type in leads:
        key = _contact_identity_key(contact_type, contact_value)
        if key:
            return key

    interests = _safe_json_obj(row["interests_json"])
    lead = interests.get("_lead") if isinstance(interests.get("_lead"), dict) else {}
    key = _contact_identity_key(
        str(lead.get("contact_type") or ""),
        str(lead.get("contact_value") or ""),
    )
    if key:
        return key

    session_id = str(row["session_id"] or "").strip()
    return f"session:{session_id}"


def _contact_identity_key(contact_type: str, contact_value: str) -> str:
    value = (contact_value or "").strip()
    if not value:
        return ""
    digits = re.sub(r"\D", "", value)
    if digits and ((contact_type or "").strip().lower() in {"phone", "mobile", "tel"} or len(digits) >= 6):
        return f"phone:{digits}"
    normalized = re.sub(r"\s+", "", value).lower()
    if not normalized:
        return ""
    contact_label = (contact_type or "contact").strip().lower() or "contact"
    return f"{contact_label}:{normalized}"


def _merge_customer_row(primary: AdminCustomerRow, incoming: AdminCustomerRow) -> AdminCustomerRow:
    return AdminCustomerRow(
        session_id=primary.session_id,
        display_name=_prefer_text(primary.display_name, incoming.display_name),
        org_name=_prefer_text(primary.org_name, incoming.org_name),
        role_category=_prefer_text(primary.role_category, incoming.role_category),
        contact=_merge_contact_text(primary.contact, incoming.contact),
        email=_prefer_text(primary.email, incoming.email),
        follow_up_status=_prefer_status(primary.follow_up_status, incoming.follow_up_status, "待跟进"),
        trial_account=_prefer_status(primary.trial_account, incoming.trial_account, "待发放"),
        updated_at=primary.updated_at,
    )


def _prefer_text(primary: str, incoming: str) -> str:
    current = (primary or "").strip()
    candidate = (incoming or "").strip()
    return current or candidate


def _prefer_status(primary: str, incoming: str, default: str) -> str:
    current = (primary or "").strip() or default
    candidate = (incoming or "").strip() or default
    if current != default:
        return current
    return candidate


def _merge_contact_text(primary: str, incoming: str) -> str:
    seen: set[str] = set()
    parts: List[str] = []
    for raw in [primary, incoming]:
        for part in (raw or "").split("/"):
            text = part.strip()
            if not text:
                continue
            key = re.sub(r"\s+", "", text).lower()
            if key in seen:
                continue
            seen.add(key)
            parts.append(text)
    return " / ".join(parts)


def _format_contact(leads: List[tuple[str, str, str]], interests: dict[str, Any]) -> str:
    parts: List[str] = []
    for contact_type, contact_value, _visitor_type in leads:
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


def _role_category(row: Any, leads: List[tuple[str, str, str]], interests: dict[str, Any]) -> str:
    raw = str(row["visitor_type"] or "").strip()
    if not raw:
        for _contact_type, _contact_value, visitor_type in leads:
            raw = (visitor_type or "").strip()
            if raw:
                break
    if not raw:
        lead = interests.get("_lead") if isinstance(interests.get("_lead"), dict) else {}
        raw = str(lead.get("visitor_type") or lead.get("role_category") or "").strip()
    return _visitor_type_label(raw)


def _visitor_type_label(visitor_type: str) -> str:
    vt = (visitor_type or "").strip()
    if vt == "institution_decision_maker":
        return "机构/学校负责人"
    if vt == "teacher":
        return "老师"
    if vt == "parent":
        return "家长"
    if vt == "student":
        return "学生"
    if vt == "other":
        return "其他"
    return ""


def _lead_email(interests: dict[str, Any]) -> str:
    lead = interests.get("_lead") if isinstance(interests.get("_lead"), dict) else {}
    return str(lead.get("email") or "").strip()


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
