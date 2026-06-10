from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal, Optional

import httpx

from app.core.config import settings
from app.core.logger import get_logger
from app.rag.generator import GeneratorConfigError
from app.schemas.chat_v5 import FriendV5SourceItem

logger = get_logger(__name__)


class FriendV5WebSourcesMissing(RuntimeError):
    pass


@dataclass(frozen=True)
class FriendV5StreamEvent:
    event: Literal["token", "web_sources", "search_keywords"]
    token: str = ""
    sources: list[FriendV5SourceItem] = field(default_factory=list)
    search_keywords: list[str] = field(default_factory=list)

    @classmethod
    def token(cls, value: str) -> "FriendV5StreamEvent":
        return cls(event="token", token=value)

    @classmethod
    def web_sources(cls, sources: list[FriendV5SourceItem]) -> "FriendV5StreamEvent":
        return cls(event="web_sources", sources=sources)

    @classmethod
    def search_keywords(cls, search_keywords: list[str]) -> "FriendV5StreamEvent":
        return cls(event="search_keywords", search_keywords=search_keywords)


class FriendV5Generator:
    def __init__(
        self,
        *,
        api_key: str,
        model: str = "",
        generation_url: str = "",
        search_strategy: str = "",
        max_tokens: int = 0,
        require_web_sources: bool = True,
        timeout_s: float = 60.0,
        client: Optional[Any] = None,
    ) -> None:
        self.model = (model or settings.chat_v5_model or "qwen3.7-plus").strip()
        self.api_key = (api_key or "").strip()
        self.generation_url = (generation_url or settings.chat_v5_generation_url).strip()
        self.search_strategy = (search_strategy or settings.chat_v5_search_strategy or "turbo").strip()
        self.max_tokens = int(max_tokens or settings.chat_v5_max_tokens or 900)
        self.require_web_sources = bool(require_web_sources)
        self.timeout_s = float(timeout_s)
        self._client = client

    async def stream(self, *, system_prompt: str, user_prompt: str) -> AsyncIterator[FriendV5StreamEvent]:
        if not self.api_key:
            raise GeneratorConfigError(f"缺少 DASHSCOPE_API_KEY，无法使用模型 {self.model}。")
        if not self.generation_url:
            raise GeneratorConfigError("缺少 CHAT_V5_GENERATION_URL。")

        payload = self._build_payload(system_prompt=system_prompt, user_prompt=user_prompt)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        emitted_source_keys: set[str] = set()
        emitted_keyword_keys: set[str] = set()
        emitted_any_source = False
        logged_structure_sample = False
        own_client = self._client is None
        client = self._client or httpx.AsyncClient()
        try:
            async with client.stream(
                "POST",
                self.generation_url,
                headers=headers,
                json=payload,
                timeout=self.timeout_s,
            ) as response:
                if hasattr(response, "raise_for_status"):
                    response.raise_for_status()
                async for raw_line in response.aiter_lines():
                    data = _parse_sse_json(raw_line)
                    if not data:
                        continue
                    if not logged_structure_sample:
                        logged_structure_sample = True
                        logger.info("V5 SSE 原始行前300字符: %s", raw_line[:300])
                        logger.info("V5 SSE 解析后 keys=%s", sorted(data.keys())[:15])
                    sources = _extract_web_sources(data)
                    new_sources = [
                        item
                        for item in sources
                        if _source_key(item) and _source_key(item) not in emitted_source_keys
                    ]
                    if new_sources:
                        emitted_any_source = True
                        for item in new_sources:
                            emitted_source_keys.add(_source_key(item))
                        yield FriendV5StreamEvent.web_sources(new_sources)
                    search_keywords = _extract_search_keywords(data)
                    new_keywords = [item for item in search_keywords if item and item not in emitted_keyword_keys]
                    if new_keywords:
                        emitted_keyword_keys.update(new_keywords)
                        yield FriendV5StreamEvent.search_keywords(new_keywords)
                    token = _extract_delta_text(data)
                    if token:
                        yield FriendV5StreamEvent.token(token)
        finally:
            if own_client and hasattr(client, "aclose"):
                await client.aclose()

        if self.require_web_sources and not emitted_any_source:
            logger.info("V5 OAI SSE 无 search_info，来源由 [SOURCES] 块提供")


    def _build_payload(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": True,
            "max_tokens": self.max_tokens,
            "enable_thinking": False,
            "enable_search": False,
            "search_options": {
                "enable_source": True,
                "search_strategy": self.search_strategy,
            },
        }


def _parse_sse_json(raw_line: str) -> dict[str, Any]:
    line = (raw_line or "").strip()
    if not line:
        return {}
    if line.startswith("data:"):
        line = line[5:].strip()
    if not line or line == "[DONE]":
        return {}
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _extract_delta_text(payload: dict[str, Any]) -> str:
    output = payload.get("output") if isinstance(payload.get("output"), dict) else {}
    text = output.get("text")
    if isinstance(text, str):
        return text
    choices = output.get("choices") or payload.get("choices") or []
    if isinstance(choices, list) and choices:
        choice = choices[0] if isinstance(choices[0], dict) else {}
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
        for obj in (delta, message, choice):
            content = obj.get("content") if isinstance(obj, dict) else None
            if isinstance(content, str):
                return content
    return ""


def _extract_web_sources(payload: dict[str, Any]) -> list[FriendV5SourceItem]:
    output = payload.get("output") if isinstance(payload.get("output"), dict) else {}
    search_info = payload.get("search_info") if isinstance(payload.get("search_info"), dict) else None
    if search_info is None:
        search_info = output.get("search_info") if isinstance(output.get("search_info"), dict) else {}
    if not search_info:
        search_info = output if isinstance(output, dict) and output.get("search_results") else {}
    raw_items: list[dict] = []
    for candidate in (
        search_info.get("search_results") if isinstance(search_info, dict) else [],
        search_info.get("results") if isinstance(search_info, dict) else [],
        payload.get("search_results"),
        payload.get("results"),
        payload.get("web_search_results"),
        payload.get("citations"),
        output.get("search_results") if isinstance(output, dict) else [],
        output.get("web_search_results") if isinstance(output, dict) else [],
    ):
        if isinstance(candidate, list) and candidate:
            raw_items = candidate
            break
    if not raw_items and search_info:
        logger.warning("V5 search_info 无匹配字段 keys=%s", sorted(search_info.keys())[:10])
    elif not raw_items and not search_info:
        logger.info("V5 响应中无 search_info 字段，payload keys=%s", sorted(payload.keys())[:15])
    items: list[FriendV5SourceItem] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        title = str(raw.get("title") or raw.get("name") or raw.get("site_name") or "").strip()
        url = str(raw.get("url") or raw.get("link") or "").strip() or None
        if not title and not url:
            continue
        snippet = str(raw.get("snippet") or raw.get("summary") or raw.get("content") or "").strip() or None
        index = _safe_int(raw.get("index"))
        items.append(
            FriendV5SourceItem(
                source_type="web",
                title=title or url or "联网搜索来源",
                url=url,
                snippet=snippet,
                index=index,
            )
        )
    return items


def _extract_search_keywords(payload: dict[str, Any]) -> list[str]:
    output = payload.get("output") if isinstance(payload.get("output"), dict) else {}
    search_info = payload.get("search_info") if isinstance(payload.get("search_info"), dict) else None
    if search_info is None:
        search_info = output.get("search_info") if isinstance(output.get("search_info"), dict) else {}
    candidates: list[Any] = []
    for item in (
        search_info.get("query") if isinstance(search_info, dict) else None,
        search_info.get("queries") if isinstance(search_info, dict) else None,
        search_info.get("search_query") if isinstance(search_info, dict) else None,
        search_info.get("search_queries") if isinstance(search_info, dict) else None,
        search_info.get("keywords") if isinstance(search_info, dict) else None,
        payload.get("query"),
        payload.get("queries"),
        payload.get("search_query"),
        payload.get("search_queries"),
        payload.get("keywords"),
        output.get("query") if isinstance(output, dict) else None,
        output.get("queries") if isinstance(output, dict) else None,
        output.get("search_query") if isinstance(output, dict) else None,
        output.get("search_queries") if isinstance(output, dict) else None,
        output.get("keywords") if isinstance(output, dict) else None,
    ):
        if item:
            candidates.append(item)

    keywords: list[str] = []
    seen: set[str] = set()

    def add_keyword(value: str) -> None:
        parts = _split_search_keywords(value)
        for part in parts:
            if part not in seen:
                seen.add(part)
                keywords.append(part)

    for candidate in candidates:
        if isinstance(candidate, str):
            add_keyword(candidate)
            continue
        if isinstance(candidate, dict):
            for key in ("query", "keyword", "text", "value"):
                raw = candidate.get(key)
                if isinstance(raw, str):
                    add_keyword(raw)
            continue
        if isinstance(candidate, list):
            for item in candidate:
                if isinstance(item, str):
                    add_keyword(item)
                elif isinstance(item, dict):
                    for key in ("query", "keyword", "text", "value"):
                        raw = item.get(key)
                        if isinstance(raw, str):
                            add_keyword(raw)
    return keywords


def _source_key(source: FriendV5SourceItem) -> str:
    return (source.url or source.title or "").strip()


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None


def _split_search_keywords(raw: str) -> list[str]:
    value = str(raw or "").strip()
    if not value:
        return []
    parts = re.split(r"[\n,，、;；|]+", value)
    out: list[str] = []
    seen: set[str] = set()
    for part in parts:
        item = part.strip().strip("\"'“”‘’")
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item[:120])
    return out


_URL_IN_TEXT_RE = re.compile(r"https?://[^\s)\]}>\"'，。、；;：]+")
_SOURCE_TAG_RE = re.compile(r"\[来源[:：]\s*(https?://[^\s\]\)]+)")

def extract_urls_from_text(text: str) -> list[FriendV5SourceItem]:
    """从回答正文中提取 URL 链接。优先匹配 [来源：URL] 格式，其次匹配裸 URL。"""
    seen: set[str] = set()
    items: list[FriendV5SourceItem] = []
    # 先试 [来源：URL] 格式
    for match in _SOURCE_TAG_RE.finditer(text or ""):
        url = match.group(1).rstrip(".,;:)]}>")
        if not url or url in seen:
            continue
        seen.add(url)
        items.append(
            FriendV5SourceItem(
                source_type="web",
                title=url,
                url=url,
                index=len(items) + 1,
            )
        )
        if len(items) >= 10:
            break
    # 再试裸 URL
    for match in _URL_IN_TEXT_RE.finditer(text or ""):
        url = match.group().rstrip(".,;:)]}>")
        if not url or url in seen:
            continue
        seen.add(url)
        items.append(
            FriendV5SourceItem(
                source_type="web",
                title=url,
                url=url,
                index=len(items) + 1,
            )
        )
        if len(items) >= 10:
            break
    return items
