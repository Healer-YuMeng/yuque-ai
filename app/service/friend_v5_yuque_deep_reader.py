from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from app.data.mcp_client import MCPSearchResult, YuqueMCPClient
from app.data.yuque_loader import YuqueDocument, YuqueLoader, build_yuque_doc_url
from app.schemas.chat import ChatMediaBundle
from app.schemas.chat_v5 import FriendV5SourceItem
from app.service.media_answer_orchestrator import (
    MediaAnswerOrchestrator,
    _DocContext,
    collect_media_from_doc_contexts,
)
from app.service.v4_vision_enrichment import enrich_media_bundle_with_vision


_DEEP_READ_HINT_RE = re.compile(
    r"(语雀|文档|指南|手册|正文|图文|图片|视频|截图|这篇|那篇|某篇|总结|提取|解读|阅读|《[^》]+》)"
)
_ALL_DOC_MEDIA_SCAN_LIMIT = 10_000
_PROMPT_BODY_CHAR_LIMIT = 3000
_VISION_PREFILTER_IMAGE_LIMIT = 2
_VISION_PREFILTER_VIDEO_LIMIT = 1
_VISION_FAST_PATH_MEDIA_THRESHOLD = 20


@dataclass(frozen=True)
class FriendV5YuqueDeepReadResult:
    used: bool = False
    prompt_block: str = ""
    sources: list[FriendV5SourceItem] = field(default_factory=list)
    media: ChatMediaBundle = field(default_factory=ChatMediaBundle)
    debug: dict[str, Any] = field(default_factory=dict)


def should_deep_read_yuque_doc(*, question: str, trigger_type: str) -> bool:
    if trigger_type == "scene":
        return False
    q = (question or "").strip()
    if not q:
        return False
    return bool(_DEEP_READ_HINT_RE.search(q))


class FriendV5YuqueDeepReader:
    def __init__(
        self,
        *,
        mcp_client: Optional[YuqueMCPClient],
        yuque_loader: Optional[YuqueLoader],
        scope: str = "",
        max_images: int = 4,
        max_videos: int = 1,
        media_enricher: Optional[
            Callable[..., Awaitable[tuple[ChatMediaBundle, list[str], dict[str, Any]]]]
        ] = None,
    ) -> None:
        self._mcp_client = mcp_client
        self._yuque_loader = yuque_loader
        self._scope = (scope or "").strip().strip("/")
        self._max_images = max(0, int(max_images or 0))
        self._max_videos = max(0, int(max_videos or 0))
        self._media_enricher = media_enricher or enrich_media_bundle_with_vision

    async def read(self, *, question: str) -> FriendV5YuqueDeepReadResult:
        if self._mcp_client is not None and getattr(self._mcp_client, "enabled", False):
            mcp_result = await self._read_via_mcp(question)
            if mcp_result.used:
                return mcp_result
        return await self._read_via_yuque_loader(question)

    async def read_toc_node(
        self,
        *,
        node: dict[str, Any],
        question: str,
    ) -> FriendV5YuqueDeepReadResult:
        title = str(node.get("title") or "").strip()
        url = str(node.get("url") or "").strip()
        doc_id = str(node.get("doc_id") or "").strip()
        if not doc_id:
            return await self.read(question=title or question)
        if self._mcp_client is not None and getattr(self._mcp_client, "enabled", False):
            try:
                body = await self._mcp_client.get_doc(doc_id)  # type: ignore[union-attr]
            except Exception:
                return FriendV5YuqueDeepReadResult(debug={"mode": "mcp_get_doc_by_toc_failed", "doc_id": doc_id})
            return await self._build_result(
                mode="mcp_get_doc_by_toc",
                doc_id=doc_id,
                title=title,
                url=url,
                snippet="",
                body=body,
                question=question or title,
            )
        if self._yuque_loader is None or not self._scope:
            return FriendV5YuqueDeepReadResult(debug={"mode": "toc_node_reader_missing", "doc_id": doc_id})
        try:
            doc: YuqueDocument = await self._yuque_loader.get_doc(book=self._scope, id_or_slug=doc_id)
        except Exception:
            return FriendV5YuqueDeepReadResult(debug={"mode": "yuque_get_doc_by_toc_failed", "doc_id": doc_id})
        return await self._build_result(
            mode="yuque_get_doc_by_toc",
            doc_id=doc.doc_id or doc_id,
            title=doc.title or title,
            url=doc.url or url,
            snippet="",
            body=doc.body,
            question=question or title,
        )

    async def _read_via_mcp(self, question: str) -> FriendV5YuqueDeepReadResult:
        try:
            hits = await self._mcp_client.search(question)  # type: ignore[union-attr]
        except Exception:
            return FriendV5YuqueDeepReadResult(debug={"mode": "mcp_search_failed"})
        hit = _pick_mcp_hit(hits)
        if hit is None or not (hit.doc_id or "").strip():
            return FriendV5YuqueDeepReadResult(debug={"mode": "mcp_search_empty"})
        try:
            body = await self._mcp_client.get_doc(hit.doc_id)  # type: ignore[union-attr]
        except Exception:
            return FriendV5YuqueDeepReadResult(debug={"mode": "mcp_get_doc_failed", "doc_id": hit.doc_id})
        return await self._build_result(
            mode="mcp_get_doc",
            doc_id=hit.doc_id,
            title=hit.title,
            url=hit.url,
            snippet=hit.snippet,
            body=body,
            question=question,
        )

    async def _read_via_yuque_loader(self, question: str) -> FriendV5YuqueDeepReadResult:
        if self._yuque_loader is None:
            return FriendV5YuqueDeepReadResult(debug={"mode": "yuque_loader_missing"})
        try:
            hits = await self._yuque_loader.search_docs(question)
        except Exception:
            return FriendV5YuqueDeepReadResult(debug={"mode": "yuque_search_failed"})
        if not hits:
            return FriendV5YuqueDeepReadResult(debug={"mode": "yuque_search_empty"})
        hit = hits[0]
        book = hit.book_id if hit.book_id is not None else self._scope
        identifier = str(hit.doc_id or hit.slug or "").strip()
        if not book or not identifier:
            return FriendV5YuqueDeepReadResult(debug={"mode": "yuque_hit_missing_identifier"})
        try:
            doc: YuqueDocument = await self._yuque_loader.get_doc(book=book, id_or_slug=identifier)
        except Exception:
            return FriendV5YuqueDeepReadResult(debug={"mode": "yuque_get_doc_failed", "doc_id": identifier})
        return await self._build_result(
            mode="yuque_get_doc",
            doc_id=doc.doc_id or identifier,
            title=doc.title or hit.title,
            url=doc.url or hit.url,
            snippet=hit.summary,
            body=doc.body,
            question=question,
        )

    async def _build_result(
        self,
        *,
        mode: str,
        doc_id: str,
        title: str,
        url: str,
        snippet: str,
        body: str,
        question: str,
    ) -> FriendV5YuqueDeepReadResult:
        body_text = (body or "").strip()
        plain = MediaAnswerOrchestrator._to_plain_text(body_text)
        if not plain and not body_text:
            return FriendV5YuqueDeepReadResult(debug={"mode": mode, "empty_body": True})
        absolute_url = build_yuque_doc_url(url, scope=self._scope)
        doc = _DocContext(
            doc_id=str(doc_id or ""),
            title=(title or "语雀文档").strip(),
            url=absolute_url,
            snippet=(snippet or "").strip(),
            body=body_text,
        )
        candidate_media = collect_media_from_doc_contexts(
            [doc],
            question=plain[:120] or doc.title,
            max_images=_ALL_DOC_MEDIA_SCAN_LIMIT,
            max_videos=_ALL_DOC_MEDIA_SCAN_LIMIT,
            primary_doc_title=doc.title,
        )
        candidate_media = _filter_low_value_media(candidate_media)
        raw_candidate_image_count = len(candidate_media.images)
        raw_candidate_video_count = len(candidate_media.videos)
        candidate_media = _prefilter_media_for_vision(
            candidate_media,
            question=question or doc.title,
            max_images=min(max(0, self._max_images), _VISION_PREFILTER_IMAGE_LIMIT),
            max_videos=min(max(0, self._max_videos), _VISION_PREFILTER_VIDEO_LIMIT),
        )
        image_limit, video_limit = _display_media_limits(candidate_media, max_images=self._max_images, max_videos=self._max_videos)
        if raw_candidate_image_count + raw_candidate_video_count > _VISION_FAST_PATH_MEDIA_THRESHOLD:
            media = ChatMediaBundle(
                images=list(candidate_media.images[:image_limit]),
                videos=list(candidate_media.videos[:video_limit]),
            )
            vision_lines: list[str] = []
            vision_debug = {
                "vision_media_skipped": "too_many_media_fast_path",
                "vision_candidate_images": len(candidate_media.images),
                "vision_candidate_videos": len(candidate_media.videos),
                "vision_display_images": len(media.images),
                "vision_display_videos": len(media.videos),
            }
        else:
            media, vision_lines, vision_debug = await self._media_enricher(
                candidate_media,
                question=question or doc.title,
                max_images=image_limit,
                max_videos=video_limit,
            )
        vision_block = "\n".join(line for line in vision_lines if str(line or "").strip()).strip()
        prompt_block = (
            "【语雀文档深读】\n"
            f"标题：{doc.title}\n"
            f"链接：{doc.url or '（无）'}\n"
            f"正文摘录：\n{plain[:_PROMPT_BODY_CHAR_LIMIT] or body_text[:_PROMPT_BODY_CHAR_LIMIT]}\n"
            f"{chr(10) + vision_block + chr(10) if vision_block else ''}"
            "请严格基于这篇语雀文档的正文和媒体信息回答；如果正文没有提到，不要编造。"
        )
        source = FriendV5SourceItem(
            source_type="yuque",
            title=doc.title,
            url=doc.url or None,
            snippet=(plain or doc.snippet)[:240] or None,
            doc_id=doc.doc_id or None,
        )
        return FriendV5YuqueDeepReadResult(
            used=True,
            prompt_block=prompt_block,
            sources=[source],
            media=media,
            debug={
                "mode": mode,
                "doc_count": 1,
                "doc_id": doc.doc_id,
                "body_chars": len(body_text),
                "candidate_media_images": raw_candidate_image_count,
                "candidate_media_videos": raw_candidate_video_count,
                "vision_prefilter_images": len(candidate_media.images),
                "vision_prefilter_videos": len(candidate_media.videos),
                "media_images": len(media.images),
                "media_videos": len(media.videos),
                **vision_debug,
            },
        )


def _pick_mcp_hit(hits: list[MCPSearchResult]) -> Optional[MCPSearchResult]:
    for hit in hits or []:
        if (hit.doc_id or "").strip():
            return hit
    return None


def _display_media_limits(media: ChatMediaBundle, *, max_images: int, max_videos: int) -> tuple[int, int]:
    video_limit = min(len(media.videos), max(0, int(max_videos or 0)), 1)
    if video_limit > 0:
        image_limit = min(len(media.images), max(0, int(max_images or 0)), 2)
    else:
        image_limit = min(len(media.images), max(0, int(max_images or 0)), 3)
    return image_limit, video_limit


def _filter_low_value_media(media: ChatMediaBundle) -> ChatMediaBundle:
    return ChatMediaBundle(
        images=[item for item in media.images if not _is_low_value_media(item.url, item.title, item.summary)],
        videos=[item for item in media.videos if not _is_low_value_media(item.url, item.title, item.summary)],
    )


def _prefilter_media_for_vision(
    media: ChatMediaBundle,
    *,
    question: str,
    max_images: int,
    max_videos: int,
) -> ChatMediaBundle:
    return ChatMediaBundle(
        images=_rank_media_for_prefilter(media.images, question=question)[: max(0, int(max_images or 0))],
        videos=_rank_media_for_prefilter(media.videos, question=question)[: max(0, int(max_videos or 0))],
    )


def _rank_media_for_prefilter(items: list[Any], *, question: str) -> list[Any]:
    if len(items) <= 1:
        return list(items)
    q_tokens = _prefilter_question_tokens(question)
    scored: list[tuple[int, int, Any]] = []
    for idx, item in enumerate(items):
        haystack = " ".join(
            str(part or "").lower()
            for part in (
                getattr(item, "title", ""),
                getattr(item, "doc_title", ""),
                getattr(item, "summary", ""),
                getattr(item, "url", ""),
            )
        )
        score = max(0, 8 - min(idx, 8))
        for token in q_tokens:
            token = (token or "").lower()
            if token and token in haystack:
                score += 8 if len(token) >= 2 else 3
        scored.append((score, -idx, item))
    scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return [item for _, _, item in scored]


def _prefilter_question_tokens(question: str) -> set[str]:
    text = (question or "").strip()
    tokens = {token for token in MediaAnswerOrchestrator._extract_keywords(text) if token}
    for term in (
        "教师支持",
        "学习软件",
        "课程目标",
        "课堂活动",
        "教学效果",
        "教师",
        "老师",
        "软件",
        "平台",
        "课程",
        "课堂",
        "作品",
        "效果",
        "培训",
        "硬件",
        "机器人",
    ):
        if term in text:
            tokens.add(term)
    return tokens


def _is_low_value_media(*parts: str) -> bool:
    text = " ".join(str(part or "").lower() for part in parts)
    return any(
        marker in text
        for marker in (
            "二维码",
            "qrcode",
            "qr.",
            "/qr",
            "头像",
            "avatar",
            "logo",
            "icon",
            "图标",
        )
    )
