from __future__ import annotations

from typing import Optional

from app.db.profile_repository import ChatSessionProfile


def display_name_for_chat(profile: Optional[ChatSessionProfile]) -> str:
    """访客可见称呼：优先完整姓名/张老师/赵先生，避免只剩姓氏「赵」。"""
    raw = (profile.display_name if profile else "") or ""
    n = raw.strip()
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
