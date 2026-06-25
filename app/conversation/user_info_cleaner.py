from __future__ import annotations

import re
from typing import Any, Optional


_EMAIL_EXTRACT_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
_EMAIL_RAW_PATTERNS = (
    re.compile(r"(?:邮箱|邮件|email|e-mail)\s*(?:是|为)?\s*[:：]?\s*([A-Z0-9._%+-@]{2,80})", re.IGNORECASE),
)

_INVALID_DISPLAY_NAMES = {
    "助手",
    "招生助手",
    "机器人",
    "客服",
    "用户",
    "本人",
    "家长",
    "爸爸",
    "妈妈",
    "父亲",
    "母亲",
}

_INVALID_ORGANIZATION_PHRASES = (
    "以后再说",
    "暂时不方便",
    "不想透露",
    "不告诉你",
    "这个不重要",
    "无",
    "没有",
    "不填",
    "不知道",
    "随便",
)


def _to_text(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def normalize_display_name_candidate(value: Any) -> Optional[str]:
    """清洗用户称呼，只保留称呼本身，不保留“我的名字是”等外壳。"""
    candidate = _to_text(value)
    if not candidate:
        return None

    candidate = re.sub(
        r"^(?:我的名字是|我名字是|名字是|姓名是|姓名|我叫|我是|叫我|称呼我|称呼)\s*[:：]?\s*",
        "",
        candidate,
        flags=re.IGNORECASE,
    ).strip()
    candidate = candidate.strip("，,。；;：: ")

    if not candidate or len(candidate) < 2 or len(candidate) > 20:
        return None
    if candidate.lower() in _INVALID_DISPLAY_NAMES:
        return None
    if re.search(r"\d", candidate):
        return None
    if re.search(r"[，。！？、,:;@/\\]", candidate):
        return None
    if not re.fullmatch(r"[A-Za-z一-龥·\s]{2,20}", candidate):
        return None
    return candidate


def normalize_organization_candidate(value: Any) -> Optional[str]:
    """清洗单位名称，去掉“单位是/我在/是”等口语前后缀。"""
    candidate = _to_text(value)
    if not candidate:
        return None

    candidate = re.sub(
        r"^(?:我的)?(?:所在)?(?:单位|公司|机构|学校)\s*(?:是|叫)?\s*[:：]?\s*",
        "",
        candidate,
        flags=re.IGNORECASE,
    )
    candidate = re.sub(r"^(?:我在|我于|我来自|来自)\s*[:：]?\s*", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\s*(?:上班|工作|任职|就职)$", "", candidate, flags=re.IGNORECASE)
    candidate = candidate.strip("，,。；;：: ")

    # 用户常输入“是xx”来回答“单位是什么”，这里的“是”不是单位名的一部分。
    candidate = re.sub(r"^是+", "", candidate).strip()

    if not candidate or len(candidate) < 2 or len(candidate) > 80:
        return None
    if any(phrase in candidate for phrase in _INVALID_ORGANIZATION_PHRASES):
        return None
    if len(candidate) > 30 and re.search(r"[，。！？,\.?!]", candidate):
        return None
    if not re.fullmatch(r"[A-Za-z0-9一-龥·（）()\-\s&]{2,80}", candidate):
        return None
    return candidate


def normalize_email_candidate(value: Any) -> Optional[str]:
    text = _to_text(value)
    if not text:
        return None
    match = _EMAIL_EXTRACT_RE.search(text)
    if not match:
        return None
    return match.group(0).strip().lower()


def extract_email_text_candidate(value: Any) -> Optional[str]:
    text = _to_text(value)
    if not text:
        return None
    normalized = normalize_email_candidate(text)
    if normalized:
        return normalized
    for pattern in _EMAIL_RAW_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        raw = str(match.group(1) or "").strip().strip("，,。；;：: ")
        if raw:
            return raw
    return None
