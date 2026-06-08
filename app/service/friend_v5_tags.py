from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

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

# 兜底标签：目录命中不足时补齐，保证每轮都有可点的下一步入口
GENERIC_FALLBACK_TAGS = [
    "想看看使用指南？",
    "想看看案例与社区？",
    "想申请测试账号，试一试产品？",
]

# 兜底标签 -> 语雀目录标题映射：点击前两个标签时直接定位对应语雀目录
TAG_TO_TOC_TITLE: dict[str, str] = {
    "想看看使用指南？": "使用指南",
    "想看看案例与社区？": "案例与社区",
}


@dataclass(frozen=True)
class FriendV5TagParseResult:
    answer: str
    tags: List[str]
    source_urls: List[str] = field(default_factory=list)


def fallback_tags_for_scene(scene: str) -> List[str]:
    # 兜底标签与场景无关，统一返回通用入口
    return list(GENERIC_FALLBACK_TAGS)


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
        return "".join(visible_parts)

    def finish(self) -> FriendV5TagParseResult:
        if self._buffer:
            if self._inside_tags:
                self._tag_parts.append(self._buffer)
            elif self._inside_sources:
                self._source_parts.append(self._buffer)
            else:
                self._answer_parts.append(self._buffer)
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


_PUNCT_ONLY_LINE_RE = re.compile(r"^[\s:：。.、，,;；!！?？\-—_*·•]+$")


def _clean_answer(answer: str) -> str:
    text = answer or ""
    # 成对隐藏块
    text = re.sub(r"\[SOURCES\].*?\[/SOURCES\]", "", text, flags=re.S)
    text = re.sub(r"\[TAGS\].*?\[END_TAGS\]", "", text, flags=re.S)
    # 落单/残留的隐藏块标记（模型偶尔只输出半个标记或回声模板）
    text = re.sub(r"\[/?SOURCES\]", "", text)
    text = re.sub(r"\[/?(?:TAGS|END_TAGS)\]", "", text)
    # markdown 链接只保留文字、去掉裸链接
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1", text)
    text = re.sub(r"https?://[^\s)\]}>\"'，。、；;：]+", "", text)
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
    return text.strip()


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
