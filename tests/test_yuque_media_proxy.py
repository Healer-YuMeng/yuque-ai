from __future__ import annotations

from app.schemas.chat import ChatMediaBundle, MediaItem
from app.service.media_answer_orchestrator import apply_yuque_proxy_to_media


def test_apply_yuque_proxy_rewrites_yuque_cdn_url() -> None:
    raw = "https://cdn.nlark.com/yuque/0/2024/png/12345/test.png"
    media = apply_yuque_proxy_to_media(
        ChatMediaBundle(images=[MediaItem(url=raw, title="t", doc_title="d")], videos=[])
    )
    assert len(media.images) == 1
    assert media.images[0].url.startswith("/yuque/asset?t=")
