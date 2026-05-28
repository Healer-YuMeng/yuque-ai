from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from app.core.config import settings
from app.core.logger import get_logger
from app.data.yuque_images import extract_image_refs_from_body, is_allowed_yuque_image_url
from app.rag.doc_image_enrichment import _download_image, _vision_caption
from app.service.media_answer_orchestrator import _DocContext

logger = get_logger(__name__)

_FIGURE_PREFIX = "【文档插图识读摘要｜供你理解配图含义，勿在回答中粘贴图片 URL】"


async def enrich_doc_contexts_with_vision(
    docs: Sequence[_DocContext],
    *,
    question: str,
) -> Tuple[List[str], Dict[str, Any]]:
    """为 MCP 文档正文中的插图生成识读摘要块，追加到 LLM 上下文。"""
    if not settings.vision_enabled or not (settings.vision_api_key or "").strip():
        return [], {"vision_skipped": "disabled"}
    token = (settings.yuque_token or "").strip()
    if not token:
        return [], {"vision_skipped": "no_token"}

    lines: List[str] = [_FIGURE_PREFIX, ""]
    used = 0
    for doc in docs:
        body = (doc.body or "").strip()
        if not body:
            continue
        refs = extract_image_refs_from_body(body)
        for ref in refs:
            if used >= settings.vision_max_images:
                break
            src = (ref.src or "").strip()
            if not src or not is_allowed_yuque_image_url(src):
                continue
            data, mime = await _download_image(src, token)
            if not data:
                continue
            caption = await _vision_caption(data, mime, user_hint=question)
            used += 1
            alt = (ref.alt or "").strip() or f"插图{used}"
            lines.append(f"- 《{doc.title}》{alt}：{caption or '（未能识读文字要点）'}")
        if used >= settings.vision_max_images:
            break

    if used <= 0:
        return [], {"vision_images_used": 0}
    return lines, {"vision_images_used": used, "vision_model": settings.vision_model}
