from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


_PHONE_RE = re.compile(r"(?<!\d)(1[3-9]\d{9})(?!\d)")

# 常见「微信号」样式；宽松兜底：取「微信是/：」后的非空白片段
_WECHAT_LABEL_RE = re.compile(
    r"(?:微信(?:号)?|wx|wechat)\s*[:：]?\s*([^\s，,。；;]{2,32})",
    re.IGNORECASE,
)
_WECHAT_ID_RE = re.compile(r"^[a-zA-Z][-_a-zA-Z0-9]{5,19}$")


@dataclass(frozen=True)
class ContactHit:
    contact_type: str  # "phone" | "wechat"
    value: str


def extract_contact(text: str) -> Optional[ContactHit]:
    """从用户单轮输入中提取手机号或微信号（第一命中）。"""
    raw = (text or "").strip()
    if not raw:
        return None
    m = _PHONE_RE.search(raw)
    if m:
        return ContactHit(contact_type="phone", value=m.group(1))
    wm = _WECHAT_LABEL_RE.search(raw)
    if wm:
        cand = wm.group(1).strip().strip("，,。；;）)】］]")
        if cand and not _PHONE_RE.fullmatch(cand):
            if _WECHAT_ID_RE.match(cand) or 2 <= len(cand) <= 32:
                return ContactHit(contact_type="wechat", value=cand)
    return None


def looks_like_decline_followup(text: str) -> bool:
    """用户明确拒绝后续联系（用于抑制无互动留资提醒）。"""
    t = (text or "").strip().lower()
    if not t:
        return False
    needles = (
        "别联系我",
        "不要联系",
        "不用联系",
        "不需要联系",
        "不要打电话",
        "别打电话",
        "不要加微信",
        "不用了谢谢",
        "骚扰",
        "不要再发",
        "别发我",
    )
    return any(n in t for n in needles)
