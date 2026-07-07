from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Literal

from app.data.yuque_loader import strip_yuque_leaks_from_text

TAG_START = "[TAGS]"
TAG_END = "[END_TAGS]"
SOURCE_START = "[SOURCES]"
SOURCE_END = "[/SOURCES]"
_SOURCE_URL_RE = re.compile(
    r"(?:https?://|www\.)[^\s\]\[<>{}\"'，。；;：]+"
    r"|(?<![@\w.-])(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,24}(?::\d+)?(?:/[^\s\]\[<>{}\"'，。；;：]*)?"
)
_SOURCE_URL_TRAILING_CHARS = ".,;:)]}>，。、；;："
_PLACEHOLDER_HOSTS = {"example.com", "example.org", "example.net"}

FriendV5TagKind = Literal["guide", "case", "trial", "price", "unknown"]

SCENE_TO_TOC_TITLE: dict[str, str] = {
    "人工智能通识教育": "人工智能通识课程",
    "跨学科项目化学习": "跨学科项目式学习",
    "智能招生": "智能招生",
    "学校AI场景定制": "学校AI场景定制",
}

# 案例标签对外展示用更短的产品名；解析时需还原为语雀 TOC 标题。
_CASE_TAG_PRODUCT_SHORT: dict[str, str] = {
    "人工智能通识课程": "人工智能通识课",
}


def toc_title_for_scene(scene: str) -> str:
    title = SCENE_TO_TOC_TITLE.get((scene or "").strip())
    return title or (scene or "").strip() or "当前产品"


def guide_tag_for_scene(scene: str) -> str:
    title = toc_title_for_scene(scene)
    return f"想看看{title}的使用指南？"


def case_tag_product_label(scene: str) -> str:
    toc_title = toc_title_for_scene(scene)
    return _CASE_TAG_PRODUCT_SHORT.get(toc_title, toc_title)


def resolve_case_tag_product_title(label: str) -> str:
    raw = (label or "").strip()
    if not raw:
        return ""
    reverse = {short: toc for toc, short in _CASE_TAG_PRODUCT_SHORT.items()}
    if raw in reverse:
        return reverse[raw]
    for toc_title in SCENE_TO_TOC_TITLE.values():
        if raw == toc_title:
            return toc_title
    return raw


def case_tag_for_scene(scene: str) -> str:
    return f"{case_tag_product_label(scene)}的优秀案例库。"


def trial_tag_for_scene(scene: str) -> str:
    title = toc_title_for_scene(scene)
    return f"想申请测试账号，试一试{title}的产品？"


def price_tag_for_scene(scene: str) -> str:
    title = toc_title_for_scene(scene)
    return f"想要了解一下{title}产品的价格？"


def explore_product_tag_for_title(title: str) -> str:
    clean = (title or "").strip()
    return f"想了解一下{clean}产品？" if clean else ""


def subdir_explore_tag_for_title(title: str) -> str:
    """把语雀子目录标题包装成统一问句风格的推荐标签。"""
    clean = (title or "").strip()
    return f"想看看{clean}？" if clean else ""


_EXPLORE_PRODUCT_TAG_RE = re.compile(r"^想了解一下(.+?)产品？$")
_CASE_TAG_PRODUCT_RE = re.compile(r"^(.+?)的优秀案例库。$")
_GUIDE_TAG_RE = re.compile(r"^想看看(.+?)的使用指南？$")
_PRODUCT_FROM_TAG_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^想看看(.+?)的产品的使用指南？$"),
    _GUIDE_TAG_RE,
    _CASE_TAG_PRODUCT_RE,
    re.compile(r"^想看看(.+?)的产品的优秀案例库？$"),
    re.compile(r"^想申请测试账号，试一试(.+?)的产品？$"),
    re.compile(r"^想要了解一下(.+?)产品的价格？$"),
)


def try_product_title_from_tag(tag: str) -> str:
    raw = (tag or "").strip()
    if not raw:
        return ""
    for pattern in _PRODUCT_FROM_TAG_PATTERNS:
        match = pattern.match(raw)
        if match:
            title = (match.group(1) or "").strip()
            if not title:
                continue
            if pattern is _CASE_TAG_PRODUCT_RE:
                return resolve_case_tag_product_title(title)
            return title
    return ""


def product_title_from_tag(tag: str, *, scene: str) -> str:
    return try_product_title_from_tag(tag) or toc_title_for_scene(scene)


def explore_product_title_from_tag(tag: str) -> str:
    raw = (tag or "").strip()
    match = _EXPLORE_PRODUCT_TAG_RE.match(raw)
    return (match.group(1) or "").strip() if match else ""


def scene_for_toc_title(title: str) -> str:
    raw = (title or "").strip()
    if not raw:
        return ""
    raw_norm = _norm_tag(raw)
    for scene_key, toc_title in SCENE_TO_TOC_TITLE.items():
        if raw_norm in {_norm_tag(scene_key), _norm_tag(toc_title)}:
            return scene_key
    return raw


def fallback_tags_for_scene(scene: str) -> List[str]:
    return [
        guide_tag_for_scene(scene),
        case_tag_for_scene(scene),
        trial_tag_for_scene(scene),
    ]


def classify_friend_v5_tag(tag: str, *, scene: str) -> FriendV5TagKind:
    raw = (tag or "").strip()
    if not raw:
        return "unknown"
    exact_map = {
        guide_tag_for_scene(scene): "guide",
        case_tag_for_scene(scene): "case",
        trial_tag_for_scene(scene): "trial",
        price_tag_for_scene(scene): "price",
    }
    if raw in exact_map:
        return exact_map[raw]  # type: ignore[return-value]
    norm = _norm_tag(raw)
    if "申请测试账号" in norm or ("试一试" in norm and "产品" in norm):
        return "trial"
    if "价格" in norm or "报价" in norm or "费用" in norm:
        return "price"
    if "优秀案例库" in norm or "产品案例" in norm or "案例库" in norm:
        return "case"
    if "使用指南" in norm or "操作说明" in norm:
        return "guide"
    return "unknown"


@dataclass(frozen=True)
class FriendV5TagParseResult:
    answer: str
    tags: List[str]
    source_urls: List[str] = field(default_factory=list)


def _norm_tag(value: str) -> str:
    return re.sub(r"[\s「」『』《》【】\[\]（）()、，。:：;；?？!！\-_/|]+", "", str(value or "").lower())


class FriendV5TagStreamFilter:

    def __init__(self, *, scene: str) -> None:
        self._scene = scene
        self._buffer = ""
        self._answer_parts: List[str] = []
        self._tag_parts: List[str] = []
        self._source_parts: List[str] = []
        self._inside_tags = False
        self._closed_tags = False
        self._inside_sources = False
        self._closed_sources = False

    def feed(self, chunk: str) -> str:
        text = str(chunk or "")
        if not text:
            return ""
        if self._closed_tags:
            return ""
        self._buffer += text
        visible_parts: List[str] = []
        while self._buffer:
            if not self._inside_sources and not self._inside_tags:
                src_idx = self._buffer.find(SOURCE_START)
                tag_idx = self._buffer.find(TAG_START)
                if tag_idx >= 0 and (src_idx < 0 or tag_idx < src_idx):
                    self._inside_tags = True
                    visible = self._buffer[:tag_idx]
                    if visible:
                        visible_parts.append(visible)
                        self._answer_parts.append(visible)
                    self._buffer = self._buffer[tag_idx + len(TAG_START):]
                    continue
                if src_idx >= 0:
                    self._inside_sources = True
                    visible = self._buffer[:src_idx]
                    if visible:
                        visible_parts.append(visible)
                        self._answer_parts.append(visible)
                    self._buffer = self._buffer[src_idx + len(SOURCE_START):]
                    continue
                keep_tag = _partial_marker_suffix_len(self._buffer, TAG_START)
                keep_src = _partial_marker_suffix_len(self._buffer, SOURCE_START)
                keep = max(keep_tag, keep_src)
                emit_len = len(self._buffer) - keep
                if emit_len <= 0:
                    break
                visible = self._buffer[:emit_len]
                visible_parts.append(visible)
                self._answer_parts.append(visible)
                self._buffer = self._buffer[emit_len:]
                break

            if self._inside_sources:
                end_idx = self._buffer.find(SOURCE_END)
                tag_idx = self._buffer.find(TAG_START)
                first = _first_non_negative(end_idx, tag_idx)
                if first == end_idx:
                    self._source_parts.append(self._buffer[:end_idx])
                    self._buffer = self._buffer[end_idx + len(SOURCE_END):]
                    self._closed_sources = True
                    self._inside_sources = False
                    continue
                if first == tag_idx:
                    self._source_parts.append(self._buffer[:tag_idx])
                    self._buffer = self._buffer[tag_idx + len(TAG_START):]
                    self._closed_sources = True
                    self._inside_sources = False
                    self._inside_tags = True
                    continue
                keep = _partial_marker_suffix_len(self._buffer, SOURCE_END)
                keep2 = _partial_marker_suffix_len(self._buffer, TAG_START)
                keep = max(keep, keep2)
                emit_len = len(self._buffer) - keep
                if emit_len <= 0:
                    break
                self._source_parts.append(self._buffer[:emit_len])
                self._buffer = self._buffer[emit_len:]
                break

            end_idx = self._buffer.find(TAG_END)
            if end_idx >= 0:
                self._tag_parts.append(self._buffer[:end_idx])
                self._buffer = ""
                self._closed_tags = True
                break
            keep = _partial_marker_suffix_len(self._buffer, TAG_END)
            emit_len = len(self._buffer) - keep
            if emit_len <= 0:
                break
            self._tag_parts.append(self._buffer[:emit_len])
            self._buffer = self._buffer[emit_len:]
            break
        return _strip_hidden_marker_fragments("".join(visible_parts))

    def finish(self) -> FriendV5TagParseResult:
        if self._buffer:
            if self._inside_tags:
                self._tag_parts.append(self._buffer)
            elif self._inside_sources:
                self._source_parts.append(self._buffer)
            else:
                leftover = _strip_trailing_marker_suffix(self._buffer)
                if leftover:
                    self._answer_parts.append(leftover)
            self._buffer = ""
        answer = _clean_answer("".join(self._answer_parts))
        tags = _parse_tags("".join(self._tag_parts))
        for item in fallback_tags_for_scene(self._scene):
            if len(tags) >= 3:
                break
            if item not in tags:
                tags.append(item)
        source_urls = _parse_source_urls("".join(self._source_parts))
        return FriendV5TagParseResult(answer=answer, tags=tags[:3], source_urls=source_urls)


_PUNCT_ONLY_LINE_RE = re.compile(r"^[\s/:：。.、，,;；!！?？\-—_*·•]+$")
# 隐藏块标记的完整或残缺片段（如流式截断后的 SOURCES]、[/SOURCES）
_HIDDEN_MARKER_FRAGMENT_RE = re.compile(
    r"(?:"
    r"\[/SOURCES\]|\[SOURCES\]|"
    r"\[/?SOURCES(?=\]|$)|"
    r"(?<!\[)(?:\[/SOURCES\]|SOURCES?\]?|OURCES?\]?|URCES?\]?|RCES?\]?|CES?\]?|ES?\]?|S\])|"
    r"\[/SOURCES(?!\])|"
    r"\[TAGS\]|\[END_TAGS\]|"
    r"\[/?(?:TAGS|END_TAGS)(?=\]|$)|"
    r"(?<!\[)(?:TAGS|END_TAGS)\]"
    r")"
)
_MARKER_SUFFIX_CANDIDATES = (SOURCE_START, SOURCE_END, TAG_START, TAG_END)


def _strip_trailing_marker_suffix(text: str) -> str:
    """去掉末尾因流式截断残留的半段标记（如 ``[S``、``SOURCES]``）。"""
    out = text or ""
    while out:
        stripped = False
        for marker in _MARKER_SUFFIX_CANDIDATES:
            max_len = min(len(out), len(marker) - 1)
            for size in range(max_len, 0, -1):
                if marker.startswith(out[-size:]):
                    out = out[:-size]
                    stripped = True
                    break
            if stripped:
                break
        if not stripped:
            break
    return out


def _strip_hidden_marker_fragments(text: str) -> str:
    out = _HIDDEN_MARKER_FRAGMENT_RE.sub("", text or "")
    return _strip_trailing_marker_suffix(out)


def _clean_answer(answer: str) -> str:
    text = answer or ""
    # 成对隐藏块
    text = re.sub(r"\[SOURCES\].*?\[/SOURCES\]", "", text, flags=re.S)
    text = re.sub(r"\[TAGS\].*?\[END_TAGS\]", "", text, flags=re.S)
    # 落单/残留的隐藏块标记（含 SOURCES] 等半段）
    text = _strip_hidden_marker_fragments(text)
    # markdown 链接只保留文字、去掉裸链接
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1", text)
    text = re.sub(r"https?://[^\s)\]}>\"'，。、；;：]+", "", text)
    text = strip_yuque_leaks_from_text(text)
    text = re.sub(r"(?m)^[ \t]*://[^\s]+[ \t]*$", "", text)
    text = re.sub(r"https?://\s*$", "", text)
    text = re.sub(r"://\s*$", "", text)
    text = re.sub(r"(?m)^[ \t]*(?:://|https?://)[ \t]*$", "", text)
    # 脚注标记（连同前后多余空格一起去掉）
    text = re.sub(r"[ \t]*\[\^\d+\][ \t]*", "", text)
    # 模型回声的提示词占位文案
    text = re.sub(r"更多正文\.{2,}", "", text)
    text = re.sub(r"(?m)^[ \t]*正文\.{2,}[ \t]*", "", text)
    # 去掉仅由标点组成的残留行（如落单的 ":" "："），但保留空行以维持分段
    kept_lines: List[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped and _PUNCT_ONLY_LINE_RE.match(stripped):
            continue
        kept_lines.append(line)
    text = "\n".join(kept_lines)
    # 中文标点前的多余空格
    text = re.sub(r"[ \t]+([，。、；：！？）])", r"\1", text)
    # 折叠多余空行
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = _fix_product_terminology(text)
    text = _strip_trial_account_disclosure(text)
    return text.strip()


_TRIAL_VERIFY_OK_RE = re.compile(r"信息校验通过[，,]?\s*已为您分配测试账号。?")
_TRIAL_ACCOUNT_BLOCK_RE = re.compile(r"【测试账号】[^\n]*")
_TRIAL_ACCOUNT_LINE_RE = re.compile(r"(?m)^[ \t]*(?:账号|密码|说明)[：:][^\n]*\s*$")


def _strip_trial_account_disclosure(text: str) -> str:
    out = text or ""
    out = _TRIAL_VERIFY_OK_RE.sub("提交成功，我们会尽快与您联系。", out)
    out = _TRIAL_ACCOUNT_BLOCK_RE.sub("", out)
    out = _TRIAL_ACCOUNT_LINE_RE.sub("", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


_APPLE_STAM_TYPO_RE = re.compile(r"苹果\s*STAM(?!E)", re.IGNORECASE)
_IDAS_PBL_TYPO_RE = re.compile(r"(?<!E)IDAS\s*-?\s*PBL", re.IGNORECASE)
_IDEAS_PBL_SPACE_RE = re.compile(r"IDEAS\s+PBL", re.IGNORECASE)


def _fix_product_terminology(text: str) -> str:
    """纠正常见产品术语笔误。"""
    out = text or ""
    out = _APPLE_STAM_TYPO_RE.sub("苹果 STEAM", out)
    out = _IDAS_PBL_TYPO_RE.sub("IDEAS-PBL", out)
    out = _IDEAS_PBL_SPACE_RE.sub("IDEAS-PBL", out)
    return out


def _parse_tags(raw: str) -> List[str]:
    if not raw.strip():
        return []
    parts = re.split(r"[\n,，；;]+", raw)
    tags: List[str] = []
    for part in parts:
        tag = part.strip().strip("-•* ")
        if not tag:
            continue
        if tag in tags:
            continue
        tags.append(tag[:60])
        if len(tags) >= 3:
            break
    return tags


def _parse_source_urls(raw: str) -> List[str]:
    if not raw.strip():
        return []
    text = raw.replace(SOURCE_START, " ").replace(SOURCE_END, " ")
    urls: List[str] = []
    for match in _SOURCE_URL_RE.finditer(text):
        url = _normalize_source_url(match.group())
        if url and url not in urls:
            urls.append(url)
    return urls


def _normalize_source_url(raw: str) -> str:
    value = (raw or "").strip().strip("<>").rstrip(_SOURCE_URL_TRAILING_CHARS)
    if not value:
        return ""
    if value.startswith("www."):
        value = f"https://{value}"
    elif not value.startswith(("http://", "https://")):
        value = f"https://{value}"
    host = re.sub(r"^https?://", "", value, flags=re.I).split("/", 1)[0].split(":", 1)[0].lower()
    if not _is_valid_source_host(host):
        return ""
    return value


def _is_valid_source_host(host: str) -> bool:
    """过滤大模型在 [SOURCES] 里编造的无效域名（如 ``ww``、占位域名）。"""
    host = (host or "").strip().strip(".")
    if not host or host in _PLACEHOLDER_HOSTS:
        return False
    if "." not in host:
        return False
    tld = host.rsplit(".", 1)[-1]
    if len(tld) < 2 or not tld.isalpha():
        return False
    return True


def _partial_marker_suffix_len(buffer: str, marker: str) -> int:
    max_len = min(len(buffer), len(marker) - 1)
    for size in range(max_len, 0, -1):
        if marker.startswith(buffer[-size:]):
            return size
    return 0


def _first_non_negative(a: int, b: int) -> int:
    if a >= 0:
        return a
    if b >= 0:
        return b
    return -1
