from __future__ import annotations

import base64
import mimetypes
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import httpx
from openai import AsyncOpenAI

from app.core.config import settings
from app.core.logger import get_logger
from app.data.yuque_images import (
    YuqueImageRef,
    encode_image_proxy_token,
    extract_image_refs_from_body,
    is_allowed_yuque_image_url,
)
from app.data.yuque_loader import YuqueDocument, YuqueLoader, YuqueLoaderError
from app.rag.retriever import RetrievalResult
from app.schemas.chat import SourceItem

logger = get_logger(__name__)

_FIGURE_BLOCK_PREFIX = "【文档插图｜已由多模态模型识读，回答请在合适段落插入下列 Markdown 图片语法，勿改写路径】"

_FIGURE_BLOCK_MARKDOWN_ONLY = (
    "【文档插图｜以下为命中文档内的图片（已映射为站内代理地址），回答请在合适段落原样插入下列 Markdown，勿改写路径】"
)


def _title_relevance_score(question: str, title: str) -> int:
    """用于多来源检索时选出与用户问题最相关的文档标题（插图只跟该篇）。"""
    q_raw = (question or "").strip()
    t = (title or "").strip()
    if not q_raw or not t:
        return 0
    q_compact = re.sub(r"\s+", "", q_raw)
    t_compact = re.sub(r"\s+", "", t)
    score = 0
    if t_compact in q_compact or q_compact in t_compact:
        score += 12
    for part in re.split(r"[\s,，。？?!！、；：]+", q_raw):
        p = part.strip()
        if len(p) < 2:
            continue
        if p in t:
            score += 3
    return score


@dataclass(frozen=True)
class _DocImageContextPick:
    """插图用 (context, source) 列表；misaligned 时与 retrieval 长度不一致，走旧逻辑。"""

    pairs: List[Tuple[str, SourceItem]]
    misaligned: bool
    multi_rejected_no_overlap: bool = False


def _pairs_for_doc_images(retrieval: RetrievalResult, question: str) -> _DocImageContextPick:
    ctxs = retrieval.contexts or []
    srcs = retrieval.sources or []
    if not ctxs or not srcs:
        return _DocImageContextPick([], False, False)
    if len(ctxs) != len(srcs):
        return _DocImageContextPick([], True, False)
    if len(srcs) == 1:
        return _DocImageContextPick([(str(ctxs[0]), srcs[0])], False, False)
    q = (question or "").strip()
    if not q:
        logger.info("doc_image_multi_source_empty_question_take_first titles=%r", [s.title for s in srcs])
        return _DocImageContextPick([(str(ctxs[0]), srcs[0])], False, False)
    scores = [_title_relevance_score(q, s.title or "") for s in srcs]
    best_i = max(range(len(scores)), key=lambda i: scores[i])
    if scores[best_i] < 1:
        logger.info(
            "doc_image_multi_source_skip_no_title_overlap scores=%s titles=%r",
            scores,
            [s.title for s in srcs],
        )
        return _DocImageContextPick([], False, True)
    logger.info(
        "doc_image_multi_source_pick_one idx=%d score=%d title=%r",
        best_i,
        scores[best_i],
        srcs[best_i].title,
    )
    return _DocImageContextPick([(str(ctxs[best_i]), srcs[best_i])], False, False)


def _ordered_keys_for_sources(srcs: List[SourceItem], book_default: str) -> List[Tuple[str, str]]:
    ordered_keys: List[Tuple[str, str]] = []
    seen_keys: Set[Tuple[str, str]] = set()
    for src in srcs:
        key = _source_fetch_keys(src, book_default)
        if key and key not in seen_keys:
            seen_keys.add(key)
            ordered_keys.append(key)
    return ordered_keys


def _parse_yuque_doc_location(url: str) -> Optional[Tuple[str, str]]:
    """从语雀文档 URL 解析 (知识库 login/repo, 文档 slug 或 id)。"""
    raw = (url or "").strip()
    if "yuque.com" not in raw:
        return None
    try:
        path = (urlparse(raw).path or "").strip("/")
    except ValueError:
        return None
    parts = [p for p in path.split("/") if p]
    if len(parts) < 3:
        return None
    return f"{parts[0]}/{parts[1]}", parts[2]


def _source_fetch_keys(source: SourceItem, default_book: str) -> Optional[Tuple[str, str]]:
    if source.doc_id and default_book and "/" in default_book:
        return (default_book, source.doc_id.strip())
    loc = _parse_yuque_doc_location(source.url or "")
    if loc:
        return loc
    return None


def _guess_image_mime(url: str, content_type: Optional[str]) -> str:
    if content_type:
        ct = content_type.split(";")[0].strip().lower()
        if ct.startswith("image/"):
            return ct
    guess, _ = mimetypes.guess_type(url)
    if guess and guess.startswith("image/"):
        return guess
    return "image/jpeg"


async def _download_image(url: str, token: str) -> tuple[Optional[bytes], Optional[str]]:
    if not is_allowed_yuque_image_url(url):
        return None, None
    headers = {
        "X-Auth-Token": token,
        "User-Agent": "enterprise-rag-mvp/0.1",
        "Referer": "https://www.yuque.com/",
        "Accept": "image/*,*/*;q=0.8",
    }
    try:
        async with httpx.AsyncClient(timeout=settings.yuque_timeout_s, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code != 200:
                logger.info("yuque_image_download_http status=%s url=%r", resp.status_code, url[:80])
                return None, None
            body = resp.content
            if len(body) > settings.vision_max_bytes:
                logger.info("yuque_image_download_too_large bytes=%d", len(body))
                return None, None
            mime = _guess_image_mime(url, resp.headers.get("content-type"))
            if not mime.startswith("image/"):
                mime = "image/jpeg"
            return body, mime
    except httpx.HTTPError as exc:
        logger.warning("yuque_image_download_failed err=%s", exc)
        return None, None


async def _vision_caption(image_bytes: bytes, mime: str, *, user_hint: str) -> str:
    key = (settings.vision_api_key or "").strip()
    if not key:
        return ""
    vb = (settings.vision_base_url or "").strip()
    client = AsyncOpenAI(api_key=key, base_url=vb or None)
    b64 = base64.standard_b64encode(image_bytes).decode("ascii")
    data_url = f"data:{mime};base64,{b64}"
    prompt = (
        "你是文档插图识读助手。请用中文简洁列出图中可见的文字要点（如有表格/列表请概括），"
        "不超过 280 字。不要编造图中没有的内容。"
        f"\n用户问题供对齐重点：{user_hint[:400]}"
    )
    try:
        resp = await client.chat.completions.create(
            model=settings.vision_model,
            temperature=0.1,
            extra_body={"enable_thinking": False},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            max_tokens=400,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as exc:  # pragma: no cover - network
        logger.warning("vision_caption_failed err=%s", exc)
        return ""


async def _vision_video_caption(video_url: str, *, user_hint: str) -> str:
    key = (settings.vision_api_key or "").strip()
    if not key:
        return ""
    vb = (settings.vision_base_url or "").strip()
    client = AsyncOpenAI(api_key=key, base_url=vb or None)
    prompt = (
        "你是文档视频识读助手。请用中文简洁概括视频里可见的核心内容、步骤或演示重点，"
        "不超过 280 字。不要编造视频中没有的内容。"
        f"\n用户问题供对齐重点：{user_hint[:400]}"
    )
    try:
        resp = await client.chat.completions.create(
            model=settings.vision_model,
            temperature=0.1,
            extra_body={"enable_thinking": False},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {"url": video_url},
                            "fps": max(1, int(settings.vision_video_fps)),
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_tokens=400,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as exc:  # pragma: no cover - network
        logger.warning("vision_video_caption_failed err=%s", exc)
        return ""


def _bodies_from_context_chunks(
    retrieval: RetrievalResult,
    *,
    question: str = "",
) -> List[Tuple[str, str, List[YuqueImageRef]]]:
    """
    仅从本轮检索片段中抽图（与 sources 按索引对齐），避免对整篇正文 get_doc 带入无关插图。
    多来源时仅保留标题与问题最相关的一篇的片段，避免 MCP title_fallback 等多篇命中时串图。
    """
    pick = _pairs_for_doc_images(retrieval, question)
    if not pick.pairs:
        return []
    bodies: List[Tuple[str, str, List[YuqueImageRef]]] = []
    for ctx, src in pick.pairs:
        refs = extract_image_refs_from_body(ctx or "")
        if not refs:
            continue
        title = ((src.title or "").strip() or "文档")[:500]
        did = str(src.doc_id or "").strip()
        bodies.append((title, did, refs))
    return bodies


async def _bodies_from_full_documents(
    ordered_keys: List[Tuple[str, str]],
    loader: YuqueLoader,
) -> List[Tuple[str, str, List[YuqueImageRef]]]:
    bodies: List[Tuple[str, str, List[YuqueImageRef]]] = []
    for book, doc_slug in ordered_keys:
        try:
            doc: YuqueDocument = await loader.get_doc(book=book, id_or_slug=doc_slug)
        except YuqueLoaderError as exc:
            logger.warning("doc_images_get_doc_failed book=%r slug=%r err=%s", book, doc_slug, exc)
            continue
        refs = extract_image_refs_from_body(doc.body or "")
        if refs:
            bodies.append((doc.title or doc_slug, str(doc.doc_id or ""), refs))
        elif (doc.body or "").strip():
            logger.info(
                "doc_images_no_refs_extracted title=%r slug=%r body_len=%d",
                doc.title,
                doc_slug,
                len(doc.body or ""),
            )
    return bodies


def _append_figure_context(
    retrieval: RetrievalResult,
    block_lines: List[str],
    extra_debug: Dict[str, Any],
) -> RetrievalResult:
    new_debug = {**(retrieval.debug or {}), **extra_debug}
    new_contexts = list(retrieval.contexts) + ["\n".join(block_lines).strip()]
    return RetrievalResult(
        contexts=new_contexts,
        sources=list(retrieval.sources),
        fallback_used=retrieval.fallback_used,
        debug=new_debug,
    )


async def enrich_retrieval_with_doc_images(
    *,
    retrieval: RetrievalResult,
    question: str,
    loader: YuqueLoader,
) -> RetrievalResult:
    """
    拉取命中文档原始 body，抽图：
    - 若启用多模态：下载 → 识读摘要 → 追加带代理 URL 的说明块；
    - 否则（或未配 vision key）：在 DOC_IMAGES_MARKDOWN_IN_CONTEXT 开启时，仅追加 Markdown 图片代理语法，不调用识图模型。
    """
    mode = (retrieval.debug or {}).get("retrieval_mode")
    if mode in ("scope_help_direct", "stale_detector"):
        return retrieval
    if retrieval.contexts and str(retrieval.contexts[0]).startswith("【合并知识库清单"):
        return retrieval

    token = (settings.yuque_token or "").strip()
    book_default = (loader.scope or "").strip().strip("/")
    if not token or not book_default or "/" not in book_default:
        logger.info("doc_images_skip_no_scope_or_token")
        return retrieval

    vision_full = settings.vision_enabled and bool((settings.vision_api_key or "").strip())
    if not vision_full and not settings.doc_images_markdown_in_context:
        return retrieval

    pick = _pairs_for_doc_images(retrieval, question)
    if pick.misaligned:
        key_sources: List[SourceItem] = list(retrieval.sources)
    elif pick.multi_rejected_no_overlap:
        key_sources = []
    elif pick.pairs:
        key_sources = [src for _, src in pick.pairs]
    else:
        key_sources = list(retrieval.sources)

    ordered_keys: List[Tuple[str, str]] = []
    seen_keys: Set[Tuple[str, str]] = set()
    for src in key_sources:
        key = _source_fetch_keys(src, book_default)
        if key and key not in seen_keys:
            seen_keys.add(key)
            ordered_keys.append(key)

    if not ordered_keys:
        return retrieval

    bodies = _bodies_from_context_chunks(retrieval, question=question)
    refs_source = "context_chunks"
    if not bodies and settings.doc_images_full_document_fallback:
        bodies = await _bodies_from_full_documents(ordered_keys, loader)
        refs_source = "full_document"
    elif not bodies:
        logger.info(
            "doc_images_no_refs_in_contexts sources=%d contexts=%d fallback=%s",
            len(retrieval.sources),
            len(retrieval.contexts),
            settings.doc_images_full_document_fallback,
        )

    if not bodies:
        return retrieval

    if settings.vision_enabled and not (settings.vision_api_key or "").strip():
        logger.info("vision_enabled_but_no_vision_api_key")

    if vision_full:
        lines: List[str] = [_FIGURE_BLOCK_PREFIX, ""]
        used = 0
        vision_model = settings.vision_model

        for title, _doc_id, refs in bodies:
            for ref in refs:
                if used >= settings.vision_max_images:
                    break
                data, mime = await _download_image(ref.src, token)
                if not data:
                    continue
                caption = await _vision_caption(data, mime, user_hint=question)
                used += 1
                proxy_t = encode_image_proxy_token(ref.src)
                md_path = f"/yuque/asset?t={proxy_t}"
                alt = ref.alt or f"插图{used}"
                lines.append(f"- 文档「{title}」插图 {used}：{caption or '（识读无文本摘要）'}")
                lines.append(f"  插入语法（请原样复制到 ## 回答 正文的合适位置）：`![{alt}]({md_path})`")
                lines.append("")
            if used >= settings.vision_max_images:
                break

        if used > 0:
            extra_debug: Dict[str, Any] = {
                "vision_enabled": True,
                "vision_images_used": used,
                "vision_model": vision_model,
                "doc_images_markdown_only": False,
                "doc_image_refs_source": refs_source,
            }
            return _append_figure_context(retrieval, lines, extra_debug)

    if not settings.doc_images_markdown_in_context:
        return retrieval

    lines_md: List[str] = [_FIGURE_BLOCK_MARKDOWN_ONLY, ""]
    used_md = 0
    for title, _doc_id, refs in bodies:
        for ref in refs:
            if used_md >= settings.vision_max_images:
                break
            if not is_allowed_yuque_image_url(ref.src):
                continue
            used_md += 1
            proxy_t = encode_image_proxy_token(ref.src)
            md_path = f"/yuque/asset?t={proxy_t}"
            alt = (ref.alt or "").strip() or f"插图{used_md}"
            lines_md.append(f"- 文档「{title}」插图 {used_md}")
            lines_md.append(f"  插入语法（请原样复制到 ## 回答 正文的合适位置）：`![{alt}]({md_path})`")
            lines_md.append("")
        if used_md >= settings.vision_max_images:
            break

    if used_md == 0:
        return retrieval

    extra_md: Dict[str, Any] = {
        "doc_images_markdown_only": True,
        "doc_image_markdown_count": used_md,
        "doc_image_refs_source": refs_source,
    }
    if vision_full:
        extra_md["vision_enabled"] = True
        extra_md["vision_images_used"] = 0
    return _append_figure_context(retrieval, lines_md, extra_md)
