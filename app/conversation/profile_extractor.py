from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from openai import AsyncOpenAI

from app.core.config import settings
from app.conversation.visitor_profile import VisitorType, detect_visitor_type
from app.db.profile_repository import ChatSessionProfile
from app.db.repositories import ChatMessageRow


@dataclass(frozen=True)
class ProfileUpdate:
    display_name: Optional[str] = None
    visitor_type: Optional[VisitorType] = None
    org_name: Optional[str] = None
    interests: Optional[Dict[str, Any]] = None


_NAME_PATTERNS: Sequence[re.Pattern[str]] = (
    re.compile(r"(?:我叫|名字是|称呼我|叫我)\s*([^\s，,。；;]{1,12})"),
    # 整段称呼含老师/教师，避免只抽到姓氏「赵」
    re.compile(r"我是\s*([^\s，,。；;]{1,10}(?:老师|教师))"),
    re.compile(r"我是\s*([^\s，,。；;]{2,6}(?:先生|女士))"),
    re.compile(r"我是\s*([^\s，,。；;]{1,12})\s*(?:老师|教师|校长|家长|学生)\b"),
    # 容错：口语/错别字「我时张老师」
    re.compile(r"我[是时]\s*([^\s，,。；;]{2,8}(?:老师|教师|校长|家长|同学)?)"),
)
_ORG_PATTERNS: Sequence[re.Pattern[str]] = (
    re.compile(r"(?:来自|在)\s*([^\s，,。；;]{2,30}?(?:学校|中学|小学|幼儿园|教育集团|机构|公司))"),
    re.compile(r"(?:单位|学校)\s*[:：]?\s*([^\s，,。；;]{2,30})"),
)

_INTEREST_STOPWORDS = {
    "你们",
    "我们",
    "平台",
    "介绍",
    "了解",
    "一下",
    "怎么",
    "如何",
    "可以",
    "能否",
    "有没有",
    "是否",
    "请问",
    "老师",
    "家长",
    "学生",
    "校长",
    "机构",
}


class ProfileExtractor:
    def __init__(self) -> None:
        self._client: Optional[AsyncOpenAI] = None
        if settings.deepseek_api_key:
            self._client = AsyncOpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url or None)

    async def extract_update(
        self,
        *,
        question: str,
        history: Sequence[ChatMessageRow],
        current_profile: Optional[ChatSessionProfile],
    ) -> ProfileUpdate:
        rule = self._extract_by_rules(question=question, history=history, current_profile=current_profile)
        # 规则已抽到核心字段，直接返回
        if rule.display_name or rule.visitor_type or rule.org_name or (rule.interests and len(rule.interests) > 0):
            return rule
        # LLM 兜底（可选）：只在已配置 DeepSeek key 时执行
        if not self._client:
            return rule
        llm = await self._extract_by_llm(question=question, history=history)
        # LLM 结果与规则合并：规则优先
        return ProfileUpdate(
            display_name=rule.display_name or llm.display_name,
            visitor_type=rule.visitor_type or llm.visitor_type,
            org_name=rule.org_name or llm.org_name,
            interests=rule.interests or llm.interests,
        )

    @staticmethod
    def _extract_by_rules(
        *,
        question: str,
        history: Sequence[ChatMessageRow],
        current_profile: Optional[ChatSessionProfile],
    ) -> ProfileUpdate:
        q = (question or "").strip()
        if not q:
            return ProfileUpdate()

        vt = detect_visitor_type(q)
        visitor_type: Optional[VisitorType] = vt if vt not in ("unknown", "other") else None

        name = _pick_first_group(q, _NAME_PATTERNS)
        if name:
            name = _sanitize_name(name)

        org = _pick_first_group(q, _ORG_PATTERNS)
        if org:
            org = org.strip().strip("，,。；;")

        interests = _extract_interests(q, base=current_profile.interests if current_profile else None)

        # 当前轮明确说了新身份时，以最新自述为准（不再兜底旧值）
        # 历史兜底：若本轮没说身份但已有记录，保留已有
        if not visitor_type and current_profile and current_profile.visitor_type:
            try:
                visitor_type = current_profile.visitor_type  # type: ignore[assignment]
            except Exception:
                visitor_type = None

        # 身份更正检测："其实/不是/我是老师"覆盖旧值
        correction_patterns = (
            re.compile(r"其实我是\s*([^\s，,。；;]{1,12})\s*(?:老师|教师|家长|校长)"),
            re.compile(r"我不是.+我是\s*([^\s，,。；;]{1,10}(?:老师|教师|家长|校长))"),
        )
        for pat in correction_patterns:
            m = pat.search(q)
            if m:
                vt_corrected = detect_visitor_type(q)
                if vt_corrected not in ("unknown", "other"):
                    visitor_type = vt_corrected
                name_corrected = _pick_first_group(q, _NAME_PATTERNS)
                if name_corrected:
                    name = _sanitize_name(name_corrected)
                break

        # 历史兜底：用户曾经说过“我是老师/我叫X/来自XX”但本轮没提
        if (not name or not org or not visitor_type) and history:
            for row in reversed(history[-12:]):
                if row.role != "user":
                    continue
                t = (row.content or "").strip()
                if not t:
                    continue
                if not visitor_type:
                    vt2 = detect_visitor_type(t)
                    if vt2 not in ("unknown", "other"):
                        visitor_type = vt2
                if not name:
                    n2 = _pick_first_group(t, _NAME_PATTERNS)
                    if n2:
                        name = _sanitize_name(n2)
                if not org:
                    o2 = _pick_first_group(t, _ORG_PATTERNS)
                    if o2:
                        org = o2.strip().strip("，,。；;")
                if visitor_type and name and org:
                    break

        return ProfileUpdate(
            display_name=name or None,
            visitor_type=visitor_type,
            org_name=org or None,
            interests=interests or None,
        )

    async def _extract_by_llm(self, *, question: str, history: Sequence[ChatMessageRow]) -> ProfileUpdate:
        # 只喂很短的历史（用户侧），避免泄露与 token 膨胀
        user_lines: List[str] = []
        for row in reversed(history[-10:]):
            if row.role != "user":
                continue
            t = (row.content or "").strip()
            if not t:
                continue
            if len(t) > 180:
                t = t[:180] + "…"
            user_lines.append(t)
        user_lines.reverse()
        hist = "\n".join([f"- {x}" for x in user_lines]) if user_lines else ""

        system = (
            "你是信息抽取器。只输出 JSON，不要输出解释文字。\n"
            "从用户话语中抽取：display_name(可选)、visitor_type(枚举：institution_decision_maker/teacher/parent/student/unknown)、"
            "org_name(可选)、interests(可选，字符串数组，最多5个)。\n"
            "规则：不要编造；抽不到就给空串或 unknown；姓名不要超过6个字；org_name不要超过20个字。"
        )
        prompt = (
            f"历史用户消息（可能为空）：\n{hist or '（无）'}\n\n"
            f"本轮用户消息：\n{question}\n\n"
            "请输出 JSON：\n"
            '{"display_name":"","visitor_type":"unknown","org_name":"","interests":[]}'
        )
        try:
            resp = await self._client.chat.completions.create(  # type: ignore[union-attr]
                model=settings.llm_model or "deepseek-chat",
                temperature=0,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            )
        except Exception:
            return ProfileUpdate()
        text = (resp.choices[0].message.content or "").strip()
        data = _safe_json_dict(text)
        display_name = _sanitize_name(str(data.get("display_name") or "")) if data else ""
        org_name = str(data.get("org_name") or "").strip()
        vt = str(data.get("visitor_type") or "unknown").strip()
        interests_raw = data.get("interests") if isinstance(data, dict) else []
        interests_list: List[str] = []
        if isinstance(interests_raw, list):
            for x in interests_raw[:5]:
                s = str(x or "").strip()
                if s and s not in interests_list:
                    interests_list.append(s)
        interests = {k: {"score": 1, "source": "llm"} for k in interests_list} if interests_list else None
        visitor_type: Optional[VisitorType] = vt if vt in ("institution_decision_maker", "teacher", "parent", "student") else None
        return ProfileUpdate(
            display_name=display_name or None,
            visitor_type=visitor_type,
            org_name=(org_name[:20] if org_name else None),
            interests=interests,
        )


def _pick_first_group(text: str, patterns: Sequence[re.Pattern[str]]) -> str:
    for p in patterns:
        m = p.search(text or "")
        if m and m.group(1):
            return m.group(1).strip()
    return ""


def _sanitize_name(name: str) -> str:
    n = (name or "").strip()
    n = re.sub(r"[\"'<>《》【】（）()]", "", n).strip()
    if not n:
        return ""
    # 避免把“老师/家长”等当成名字
    if n in ("老师", "家长", "学生", "校长"):
        return ""
    return n[:6]


def _extract_interests(text: str, *, base: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    # 简单关键词抽取：中文 2-10 字片段，过滤 stopwords
    raw = re.split(r"[\s,，。！？?、]+", (text or "").strip())
    terms = []
    for x in raw:
        s = (x or "").strip()
        if len(s) < 2 or len(s) > 12:
            continue
        # stopwords 只做“整词”过滤，避免误杀“备课流程”等
        if s in _INTEREST_STOPWORDS:
            continue
        terms.append(s)
    terms = terms[:6]
    out: Dict[str, Any] = dict(base or {})
    for t in terms:
        prev = out.get(t)
        score = 1
        if isinstance(prev, dict):
            try:
                score = int(prev.get("score") or 1) + 1
            except Exception:
                score = 2
        out[t] = {"score": min(score, 5), "source": "rule"}
    return out


def _safe_json_dict(text: str) -> Dict[str, Any]:
    s = (text or "").strip()
    if not s:
        return {}
    # 容错：如果模型输出了 ```json ... ```，取中间
    if "```" in s:
        parts = s.split("```")
        for part in parts:
            if "{" in part and "}" in part:
                s = part.strip()
                break
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}

