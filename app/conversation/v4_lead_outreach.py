from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from app.conversation.contact_extractor import extract_contact
from app.conversation.lead_nudge_policy import LeadNudgeDecision, LeadNudgePolicy
from app.conversation.visitor_intent import detect_intent_flags
from app.conversation.chat_display import display_name_for_chat
from app.db.profile_repository import ChatSessionProfile
from app.db.repositories import ChatMessageRow, LeadCaptureRepository


_LEAD_INTEREST_KEY = "_lead"


@dataclass(frozen=True)
class V4LeadTurnResult:
    lead_saved: bool
    contact_detected: bool
    interests_patch: Dict[str, Any]


@dataclass(frozen=True)
class V4LeadEndResult:
    nudge: LeadNudgeDecision
    trial_apply_available: bool
    append_text: str = ""
    asked_field: str = ""


class V4LeadOutreach:
    """V4 留资：结构化采集 + 适当时机轻引导 + 试用按钮。"""

    def __init__(
        self,
        *,
        lead_policy: LeadNudgePolicy,
        lead_capture_repository: LeadCaptureRepository,
    ) -> None:
        self._lead_policy = lead_policy
        self._lead_repo = lead_capture_repository

    async def ingest_user_turn(
        self,
        *,
        session_id: str,
        question: str,
        profile: Optional[ChatSessionProfile],
        catalog_path: Sequence[str],
    ) -> V4LeadTurnResult:
        sid = (session_id or "").strip()
        q = (question or "").strip()
        patch: Dict[str, Any] = dict((profile.interests if profile else {}).get(_LEAD_INTEREST_KEY) or {})
        if not isinstance(patch, dict):
            patch = {}

        contact = extract_contact(q)
        lead_saved = False
        if contact and sid:
            vt = (profile.visitor_type if profile else "") or None
            lead_saved = await self._lead_repo.try_insert_lead(
                session_id=sid,
                contact_type=contact.contact_type,
                contact_value=contact.value,
                visitor_type=vt if vt else None,
            )
            patch["contact_type"] = contact.contact_type
            patch["contact_value"] = contact.value

        if profile and profile.org_name:
            patch["org_name"] = profile.org_name

        purchase, trial = detect_intent_flags(q)
        if trial or _mentions_trial(q):
            patch["wants_trial"] = True
        if purchase:
            patch["purchase_intent"] = True

        product = _detect_interested_product(q, catalog_path=catalog_path)
        if product:
            patch["interested_product"] = product

        interests_patch: Dict[str, Any] = {}
        if patch:
            interests_patch[_LEAD_INTEREST_KEY] = patch

        return V4LeadTurnResult(
            lead_saved=lead_saved,
            contact_detected=bool(contact),
            interests_patch=interests_patch,
        )

    async def evaluate_end_of_turn(
        self,
        *,
        session_id: str,
        question: str,
        history: Sequence[ChatMessageRow],
        profile: Optional[ChatSessionProfile],
        dialog_level: int,
        is_guide_only: bool,
    ) -> V4LeadEndResult:
        sid = (session_id or "").strip()
        has_lead = await self._lead_repo.has_lead_for_session(session_id=sid) if sid else False
        lead_meta = _lead_meta(profile)
        session_meta = _session_meta(profile)
        has_contact = has_lead or bool(lead_meta.get("contact_value"))
        user_rounds = sum(1 for row in history if row.role == "user")

        nudge = LeadNudgeDecision(triggered=False)
        # 功能讲解中（深度内容）才在收尾轻引导；纯目录引导轮次不弹留资
        if not is_guide_only and dialog_level >= 2 and user_rounds >= 4:
            nudge = self._lead_policy.decide(
                question=question,
                history=history,
                has_existing_lead=has_contact and _lead_complete(lead_meta, profile),
            )
            if nudge.triggered:
                nudge_text, asked_field = _build_v4_nudge_text(
                    profile=profile,
                    lead_meta=lead_meta,
                    session_meta=session_meta,
                )
                nudge = LeadNudgeDecision(
                    triggered=bool(nudge_text),
                    reason=nudge.reason,
                    text=nudge_text,
                )
            else:
                asked_field = ""
        else:
            asked_field = ""

        wants_trial = bool(lead_meta.get("wants_trial")) or _mentions_trial(question)
        trial_apply_available = wants_trial and not is_guide_only

        append = ""
        if nudge.triggered and nudge.text:
            append = "\n\n" + nudge.text.strip()

        return V4LeadEndResult(
            nudge=nudge,
            trial_apply_available=trial_apply_available,
            append_text=append,
            asked_field=asked_field,
        )


def _lead_meta(profile: Optional[ChatSessionProfile]) -> Dict[str, Any]:
    if not profile or not isinstance(profile.interests, dict):
        return {}
    raw = profile.interests.get(_LEAD_INTEREST_KEY)
    return raw if isinstance(raw, dict) else {}


def _session_meta(profile: Optional[ChatSessionProfile]) -> Dict[str, Any]:
    if not profile or not isinstance(profile.interests, dict):
        return {}
    raw = profile.interests.get("_session")
    return raw if isinstance(raw, dict) else {}


def _lead_complete(lead_meta: Dict[str, Any], profile: Optional[ChatSessionProfile]) -> bool:
    has_name = bool(display_name_for_chat(profile))
    has_contact = bool(lead_meta.get("contact_value"))
    has_org = bool(lead_meta.get("org_name") or (profile.org_name if profile else ""))
    return has_name and has_contact and has_org


def _build_v4_nudge_text(
    *,
    profile: Optional[ChatSessionProfile],
    lead_meta: Dict[str, Any],
    session_meta: Dict[str, Any],
) -> tuple[str, str]:
    name = display_name_for_chat(profile)
    prefix = f"{name}，" if name else ""
    asked_fields = {str(x) for x in (session_meta.get("asked_fields") or []) if str(x).strip()}
    suppressed_fields = {str(x) for x in (session_meta.get("suppressed_fields") or []) if str(x).strip()}
    blocked = asked_fields | suppressed_fields

    checks = [
        ("name", not name, "为了后续沟通更顺一点，方便告诉我该怎么称呼您吗？"),
        ("org_name", not (lead_meta.get("org_name") or (profile.org_name if profile else "")), "为了后续给您更贴合的方案，方便补充一下您的单位或学校吗？"),
        (
            "contact",
            not lead_meta.get("contact_value"),
            "我这里有一份完整案例资料，可以发给您参考。方便留一个联系方式吗？我安排顾问发给您。",
        ),
    ]
    for field, missing, text in checks:
        if missing and field not in blocked:
            return (f"{prefix}{text}" if prefix else text, field)
    return ("", "")


def _mentions_trial(q: str) -> bool:
    t = (q or "").strip()
    return any(k in t for k in ("申请测试", "试用", "体验一下", "演示账号", "测试账号", "开通试用"))


def _detect_interested_product(q: str, *, catalog_path: Sequence[str]) -> str:
    if catalog_path:
        return catalog_path[-1]
    for pat in (
        r"(?:感兴趣|想了解|看看|介绍)(?:一下)?[《「]?([^》」\s，,。；;]{2,24})",
    ):
        m = re.search(pat, q or "")
        if m and m.group(1):
            return m.group(1).strip()
    return ""
