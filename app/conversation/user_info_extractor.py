from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from app.conversation.contact_extractor import extract_contact
from app.conversation.user_info_cleaner import (
    extract_email_text_candidate,
    normalize_display_name_candidate,
    normalize_email_candidate,
    normalize_organization_candidate,
)
from app.core.config import settings


_DEFAULT_CLIENT = object()


class StructuredUserInfoSchema(BaseModel):
    display_name: Optional[str] = Field(default=None, description="用户希望被如何称呼")
    org_name: Optional[str] = Field(default=None, description="用户明确提供的单位、学校、公司或机构名称")
    contact: Optional[str] = Field(default=None, description="用户明确提供的手机号或微信号")
    email: Optional[str] = Field(default=None, description="用户明确提供的邮箱")


@dataclass(frozen=True)
class StructuredUserInfo:
    display_name: str = ""
    org_name: str = ""
    contact: str = ""
    email: str = ""


USER_INFO_EXTRACTION_SYSTEM_PROMPT = """
你是一名信息提取助手。你的任务是从用户输入或申请表文本中，提取用户明确提供的关键信息，并以结构化 JSON 返回。

提取字段：
- display_name：用户希望被如何称呼的名字、昵称或称呼
- org_name：用户明确提到的单位、公司、学校或机构名称
- contact：用户明确提供的手机号或微信号
- email：用户明确提供的邮箱

提取规则：
1. 只提取字段值，不要提取整句话。
2. 如果用户说“我叫 zjt”“我的名字是 zjt”“叫我 Lisa 就行”，display_name 只提取 zjt、zjt、Lisa。
3. display_name 不能是“助手”“客服”“家长”“爸爸”“妈妈”这类泛称或机器人称呼。
4. org_name 只保留单位名称本身，不保留“我在…上班”“单位是…”这类前缀整句。
5. contact 只有在用户明确提供手机号或微信号时才提取；不要猜测。
6. email 只有在用户明确提供邮箱时才提取；不要猜测。
7. 字段不存在或无法确定时返回 null。
8. 只输出 JSON，不要输出解释、注释、Markdown 或其他多余文字。

输出格式：
{
  "display_name": null,
  "org_name": null,
  "contact": null,
  "email": null
}
""".strip()


class UserInfoStructuredExtractor:
    def __init__(
        self,
        *,
        client: Any = _DEFAULT_CLIENT,
        model: Optional[str] = None,
    ) -> None:
        self._model = (model or settings.llm_model or "qwen3.7-plus").strip()
        if client is not _DEFAULT_CLIENT:
            self._client = client
            return
        api_key, base_url = settings.resolve_model_endpoint(self._model)
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url or None) if api_key else None

    async def extract(self, transcript: str) -> StructuredUserInfo:
        raw: Dict[str, Any] = {}
        if self._client:
            raw = await self._extract_by_llm(transcript)
        cleaned = _clean_structured_info(raw)
        return _fallback_extract_info(cleaned, transcript)

    async def _extract_by_llm(self, transcript: str) -> Dict[str, Any]:
        text = (transcript or "").strip()
        if not text:
            return {}
        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                temperature=0,
                response_format={"type": "json_object"},
                extra_body={"enable_thinking": False},
                messages=[
                    {"role": "system", "content": USER_INFO_EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
            )
        except Exception:
            return {}
        content = (resp.choices[0].message.content or "").strip()
        data = _safe_json_dict(content)
        if not data:
            return {}
        try:
            return StructuredUserInfoSchema(**data).model_dump()
        except Exception:
            return data


def _clean_structured_info(data: Dict[str, Any]) -> StructuredUserInfo:
    contact_text = str(data.get("contact") or "").strip()
    contact_hit = extract_contact(contact_text)
    return StructuredUserInfo(
        display_name=normalize_display_name_candidate(data.get("display_name")) or "",
        org_name=normalize_organization_candidate(data.get("org_name")) or "",
        contact=contact_hit.value if contact_hit else contact_text,
        email=normalize_email_candidate(data.get("email")) or "",
    )


def _fallback_extract_info(current: StructuredUserInfo, transcript: str) -> StructuredUserInfo:
    text = (transcript or "").strip()
    contact_hit = extract_contact(current.contact or text)
    return StructuredUserInfo(
        display_name=current.display_name or _extract_explicit_display_name(text) or "",
        org_name=current.org_name or _extract_explicit_org_name(text) or "",
        contact=(contact_hit.value if contact_hit else current.contact),
        email=current.email or extract_email_text_candidate(text) or "",
    )


def _extract_explicit_display_name(text: str) -> str:
    patterns = (
        r"(?:我的名字是|我名字是|名字是|姓名是|姓名|我叫|叫我|称呼我|称呼)\s*[:：]?\s*([A-Za-z一-龥· ]{2,20})",
        r"(?:我是|我时)\s*([A-Za-z一-龥· ]{1,20}(?:老师|教师|校长|主任|先生|女士|家长|同学))",
        r"(?:my name is|i am|i'm|call me)\s+([A-Za-z][A-Za-z\s]{1,19})",
    )
    for pattern in patterns:
        match = re.search(pattern, text or "", re.IGNORECASE)
        if not match:
            continue
        name = normalize_display_name_candidate(match.group(1))
        if name:
            return name
    return ""


def _extract_explicit_org_name(text: str) -> str:
    patterns = (
        r"我在\s*([A-Za-z0-9一-龥·（）()\-\s&]{2,80})\s*(?:上班|工作|任职|就职)",
        r"(?:我的单位是|所在单位是|单位是|单位叫|公司是|公司叫|机构是|学校是|来自)\s*[:：]?\s*([A-Za-z0-9一-龥·（）()\-\s&]{2,80})",
    )
    for pattern in patterns:
        match = re.search(pattern, text or "", re.IGNORECASE)
        if not match:
            continue
        org = normalize_organization_candidate(match.group(1))
        if org:
            return org
    return ""


def _safe_json_dict(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {}
    if "```" in raw:
        for part in raw.split("```"):
            if "{" in part and "}" in part:
                raw = part.strip()
                break
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        pass
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}
