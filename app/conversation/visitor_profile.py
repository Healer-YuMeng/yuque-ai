from __future__ import annotations

import re
from typing import Literal

VisitorType = Literal[
    "institution_decision_maker",
    "teacher",
    "parent",
    "student",
    "other",
    "unknown",
]


def detect_visitor_type(text: str) -> VisitorType:
    """轻量规则识别访客身份（单轮）。"""
    t = (text or "").strip()
    if not t:
        return "unknown"
    low = t.lower()

    if re.search(r"(机构|培训|学校|校长|园长|负责人|采购|校方)", t):
        return "institution_decision_maker"
    if "老师" in t or "教师" in t or "班主任" in t:
        return "teacher"
    if "家长" in t or "孩子" in t or "小孩" in t or "学生家" in t:
        return "parent"
    if "家长" not in t and (
        re.search(r"(我是|本人|在读)\s*学生", t)
        or re.search(r"^学生[。.\s]*$", t.strip())
        or ("student" in low and "家长" not in t and len(t) < 24)
    ):
        return "student"

    if any(k in t for k in ("了解一下", "随便看看", "路过", "看看")):
        return "other"

    return "unknown"
