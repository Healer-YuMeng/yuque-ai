from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Sequence, Tuple

from app.core.config import settings
from app.core.logger import get_logger
from app.data.yuque_images import extract_image_refs_from_body, is_allowed_yuque_image_url
from app.rag.doc_image_enrichment import _download_image, _vision_caption, _vision_video_caption
from app.service.media_answer_orchestrator import MediaAnswerOrchestrator, _DocContext
from app.schemas.chat import ChatMediaBundle, MediaItem

logger = get_logger(__name__)

_FIGURE_PREFIX = "【文档多媒体识读摘要｜供你理解配图和视频含义，勿在回答中粘贴媒体 URL】"
_VISION_IMAGE_SUMMARY_CACHE: Dict[tuple[str, str], str] = {}
_VISION_VIDEO_SUMMARY_CACHE: Dict[tuple[str, str], str] = {}


def _normalize_question_keywords(question: str) -> List[str]:
    kws = MediaAnswerOrchestrator._extract_keywords(question)
    return [kw.lower() for kw in kws if (kw or "").strip()]


def _media_match_score(*, question: str, item: MediaItem, summary: str, media_kind: str, original_rank: int) -> int:
    haystack = f"{item.title} {item.doc_title} {summary}".lower()
    score = max(0, 12 - original_rank)
    for kw in _normalize_question_keywords(question):
        if kw and kw in haystack:
            score += 5 if len(kw) >= 2 else 2
    q = (question or "").lower()
    if media_kind == "video" and any(k in q for k in ("视频", "演示", "录屏", "讲解", "demo")):
        score += 6
    if media_kind == "image" and any(k in q for k in ("图片", "截图", "界面", "海报", "课件")):
        score += 6
    return score


async def _get_cached_image_summary(*, src: str, token: str, question: str) -> str:
    cache_key = ((src or "").strip(), settings.vision_model)
    if not cache_key[0]:
        return ""
    cached = _VISION_IMAGE_SUMMARY_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if not is_allowed_yuque_image_url(src):
        return ""
    data, mime = await _download_image(src, token)
    if not data:
        return ""
    summary = (await _vision_caption(data, mime, user_hint=question)).strip()
    if summary:
        _VISION_IMAGE_SUMMARY_CACHE[cache_key] = summary
    return summary


async def _get_cached_video_summary(*, src: str, question: str) -> str:
    cache_key = ((src or "").strip(), settings.vision_model)
    if not cache_key[0]:
        return ""
    cached = _VISION_VIDEO_SUMMARY_CACHE.get(cache_key)
    if cached is not None:
        return cached
    summary = (await _vision_video_caption(src, user_hint=question)).strip()
    if summary:
        _VISION_VIDEO_SUMMARY_CACHE[cache_key] = summary
    return summary


async def enrich_doc_contexts_with_vision(
    docs: Sequence[_DocContext],
    *,
    question: str,
) -> Tuple[List[str], Dict[str, Any]]:
    """为 MCP 文档正文中的图片/视频生成识读摘要块，追加到 LLM 上下文。"""
    if not settings.vision_enabled or not (settings.vision_api_key or "").strip():
        return [], {"vision_skipped": "disabled"}
    token = (settings.yuque_token or "").strip()
    if not token:
        return [], {"vision_skipped": "no_token"}

    image_jobs: List[Tuple[_DocContext, str, str]] = []
    video_jobs: List[Tuple[_DocContext, str]] = []
    for doc in docs:
        body = (doc.body or "").strip()
        if not body:
            continue
        refs = extract_image_refs_from_body(body)
        for ref in refs:
            if len(image_jobs) >= settings.vision_max_images:
                break
            src = (ref.src or "").strip()
            if not src or not is_allowed_yuque_image_url(src):
                continue
            alt = (ref.alt or "").strip()
            image_jobs.append((doc, src, alt))
        for ref in MediaAnswerOrchestrator._extract_video_urls(body):
            if len(video_jobs) >= settings.vision_max_videos:
                break
            src = (ref.url or "").strip()
            if not src:
                continue
            video_jobs.append((doc, src))
        if len(image_jobs) >= settings.vision_max_images and len(video_jobs) >= settings.vision_max_videos:
            break

    image_captions = await asyncio.gather(
        *[_get_cached_image_summary(src=src, token=token, question=question) for _, src, _ in image_jobs],
        return_exceptions=True,
    )
    video_captions = await asyncio.gather(
        *[_get_cached_video_summary(src=src, question=question) for _, src in video_jobs],
        return_exceptions=True,
    )

    lines: List[str] = [_FIGURE_PREFIX, ""]
    used_images = 0
    used_videos = 0
    for idx, (doc, _, alt) in enumerate(image_jobs):
        caption = image_captions[idx]
        if isinstance(caption, Exception):
            logger.warning("vision_image_summary_failed", exc_info=caption)
            continue
        caption = (caption or "").strip()
        if not caption:
            continue
        used_images += 1
        label = alt or f"插图{used_images}"
        lines.append(f"- 参考图{used_images}｜《{doc.title}》{label}：{caption or '（未能识读文字要点）'}")

    for idx, (doc, _) in enumerate(video_jobs):
        caption = video_captions[idx]
        if isinstance(caption, Exception):
            logger.warning("vision_video_summary_failed", exc_info=caption)
            continue
        caption = (caption or "").strip()
        used_videos += 1
        lines.append(f"- 参考视频{used_videos}｜《{doc.title}》：{caption or '（未能识读视频内容）'}")

    if used_images <= 0 and used_videos <= 0:
        return [], {"vision_images_used": 0, "vision_videos_used": 0}
    return lines, {
        "vision_images_used": used_images,
        "vision_videos_used": used_videos,
        "vision_model": settings.vision_model,
    }


async def enrich_media_bundle_with_vision(
    media: ChatMediaBundle,
    *,
    question: str,
    max_images: int,
    max_videos: int,
) -> Tuple[ChatMediaBundle, List[str], Dict[str, Any]]:
    if not media.images and not media.videos:
        return ChatMediaBundle(), [], {"vision_media_skipped": "empty"}
    if not settings.vision_enabled or not (settings.vision_api_key or "").strip():
        return (
            ChatMediaBundle(
                images=list(media.images[: max(0, int(max_images))]),
                videos=list(media.videos[: max(0, int(max_videos))]),
            ),
            [],
            {"vision_media_skipped": "disabled"},
        )

    token = (settings.yuque_token or "").strip()
    enriched_images: List[tuple[int, int, MediaItem]] = []
    enriched_videos: List[tuple[int, int, MediaItem]] = []

    image_summaries = await asyncio.gather(
        *[
            _get_cached_image_summary(src=(item.url or "").strip(), token=token, question=question)
            if (item.url or "").strip() and token and is_allowed_yuque_image_url((item.url or "").strip())
            else asyncio.sleep(0, result=(item.summary or "").strip())
            for item in media.images
        ],
        return_exceptions=True,
    )
    video_summaries = await asyncio.gather(
        *[
            _get_cached_video_summary(src=(item.url or "").strip(), question=question)
            if (item.url or "").strip()
            else asyncio.sleep(0, result=(item.summary or "").strip())
            for item in media.videos
        ],
        return_exceptions=True,
    )

    for idx, item in enumerate(media.images):
        resolved = image_summaries[idx]
        if isinstance(resolved, Exception):
            logger.warning("vision_image_media_failed", exc_info=resolved)
            summary = (item.summary or "").strip()
        else:
            summary = (resolved or "").strip() or (item.summary or "").strip()
        score = _media_match_score(
            question=question,
            item=item,
            summary=summary,
            media_kind="image",
            original_rank=idx,
        )
        enriched_images.append(
            (
                score,
                idx,
                MediaItem(
                    url=item.url,
                    title=item.title,
                    doc_title=item.doc_title,
                    doc_id=item.doc_id,
                    summary=summary,
                ),
            )
        )

    for idx, item in enumerate(media.videos):
        resolved = video_summaries[idx]
        if isinstance(resolved, Exception):
            logger.warning("vision_video_media_failed", exc_info=resolved)
            summary = (item.summary or "").strip()
        else:
            summary = (resolved or "").strip() or (item.summary or "").strip()
        score = _media_match_score(
            question=question,
            item=item,
            summary=summary,
            media_kind="video",
            original_rank=idx,
        )
        enriched_videos.append(
            (
                score,
                idx,
                MediaItem(
                    url=item.url,
                    title=item.title,
                    doc_title=item.doc_title,
                    doc_id=item.doc_id,
                    summary=summary,
                ),
            )
        )

    enriched_images.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    enriched_videos.sort(key=lambda x: (x[0], -x[1]), reverse=True)

    final_images = [item for _, _, item in enriched_images[: max(0, int(max_images))]]
    final_videos = [item for _, _, item in enriched_videos[: max(0, int(max_videos))]]

    lines: List[str] = []
    if final_images or final_videos:
        lines = [_FIGURE_PREFIX, ""]
        for idx, item in enumerate(final_images, start=1):
            lines.append(
                f"- 参考图{idx}｜《{item.doc_title or item.title or '图片'}》：{(item.summary or '').strip() or '（未能识读到稳定文字或画面重点）'}"
            )
        for idx, item in enumerate(final_videos, start=1):
            lines.append(
                f"- 参考视频{idx}｜《{item.doc_title or item.title or '视频'}》：{(item.summary or '').strip() or '（未能识读到稳定视频重点）'}"
            )

    return (
        ChatMediaBundle(images=final_images, videos=final_videos),
        lines,
        {
            "vision_images_used": len(final_images),
            "vision_videos_used": len(final_videos),
            "vision_model": settings.vision_model,
        },
    )
