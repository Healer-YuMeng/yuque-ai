from __future__ import annotations

import re
from typing import Optional

from app.db.profile_repository import ChatSessionProfile

_INVALID_CHAT_DISPLAY_NAME_PATTERNS = (
    re.compile(r"^(?:低年级|中年级|高年级|低中年级|中高年级)$"),
    re.compile(r"^(?:小学|初中|高中|大学)(?:阶段|年级)?$"),
    re.compile(r"^[一二三四五六七八九十]+年级$"),
    re.compile(r"^[0-9]+年级$"),
    re.compile(r"^(?:软件项目|软件编程|硬件搭建|信息课|社团)$"),
    re.compile(r"^(?:给|带|做|看)(?:小学|初中|高中|低年级|中年级|高年级|低中年级|中高年级|软件项目|软件编程|硬件搭建|社团).*$"),
)


def display_name_for_chat(profile: Optional[ChatSessionProfile]) -> str:
    """访客可见称呼：优先完整姓名/张老师/赵先生，避免只剩姓氏「赵」。"""
    raw = (profile.display_name if profile else "") or ""
    org = (profile.org_name if profile else "") or ""
    n = _normalize_display_name(raw, org=org)
    if not n:
        return ""
    if n.endswith(("老师", "校长", "家长", "同学", "先生", "女士")):
        return n
    vt = (profile.visitor_type if profile else "") or ""
    if vt == "teacher":
        return f"{n}老师" if len(n) <= 4 else n
    if vt == "parent":
        return f"{n}先生" if len(n) <= 4 and not n.endswith("先生") else n
    return n


def normalize_display_name(raw: str, *, org: str = "") -> str:
    return _normalize_display_name(raw, org=org)


def _normalize_display_name(raw: str, *, org: str = "") -> str:
    n = (raw or "").strip()
    if not n:
        return ""
    org_clean = (org or "").strip()
    if org_clean:
        for prefix in (
            f"{org_clean}的",
            org_clean,
            f"来自{org_clean}的",
            f"在{org_clean}的",
            f"我是{org_clean}的",
            f"我时{org_clean}的",
        ):
            if n.startswith(prefix):
                n = n[len(prefix) :].strip()
                break
    if "的" in n and n.endswith(("老师", "教师", "校长", "先生", "女士", "同学", "家长", "主任", "院长", "园长")):
        tail = n.split("的")[-1].strip()
        if tail:
            n = tail
    n = re.sub(r"^(我是|我时|我叫|名字是|称呼我|叫我)", "", n).strip()
    n = n.strip("，,。；;：: ")
    if n in ("老师", "教师", "家长", "学生", "同学", "校长", "先生", "女士"):
        return ""
    if any(pat.fullmatch(n) for pat in _INVALID_CHAT_DISPLAY_NAME_PATTERNS):
        return ""
    return n
