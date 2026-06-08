from __future__ import annotations

import pytest

import app.service.v4_vision_enrichment as vision_enrichment
from app.core.config import settings
from app.data.yuque_images import encode_image_proxy_token
from app.schemas.chat import ChatMediaBundle, MediaItem


@pytest.mark.asyncio
async def test_enrich_media_bundle_decodes_yuque_asset_proxy_for_vision(monkeypatch: pytest.MonkeyPatch) -> None:
    original_enabled = settings.vision_enabled
    original_key = settings.vision_api_key
    original_model = settings.vision_model
    original_token = settings.yuque_token
    raw_url = "https://cdn.nlark.com/yuque/0/2026/png/67444281/course.png"
    proxy_url = f"/yuque/asset?t={encode_image_proxy_token(raw_url)}"
    downloaded: list[str] = []

    async def fake_download_image(src: str, token: str):
        downloaded.append(src)
        return b"image-bytes", "image/png"

    async def fake_vision_caption(image_bytes: bytes, mime: str, *, user_hint: str) -> str:
        assert image_bytes == b"image-bytes"
        assert mime == "image/png"
        assert "乐高" in user_hint
        return "图片展示了乐高课程的学习包、积木搭建和课堂协作要点。"

    monkeypatch.setattr(vision_enrichment, "_download_image", fake_download_image)
    monkeypatch.setattr(vision_enrichment, "_vision_caption", fake_vision_caption)
    vision_enrichment._VISION_IMAGE_SUMMARY_CACHE.clear()
    try:
        object.__setattr__(settings, "vision_enabled", True)
        object.__setattr__(settings, "vision_api_key", "fake-key")
        object.__setattr__(settings, "vision_model", "fake-qwen-vl")
        object.__setattr__(settings, "yuque_token", "fake-yuque-token")

        media, lines, debug = await vision_enrichment.enrich_media_bundle_with_vision(
            ChatMediaBundle(
                images=[
                    MediaItem(
                        url=proxy_url,
                        doc_title="乐高人工智能课程",
                        doc_id="101",
                    )
                ],
                videos=[],
            ),
            question="乐高人工智能课程",
            max_images=1,
            max_videos=0,
        )
    finally:
        object.__setattr__(settings, "vision_enabled", original_enabled)
        object.__setattr__(settings, "vision_api_key", original_key)
        object.__setattr__(settings, "vision_model", original_model)
        object.__setattr__(settings, "yuque_token", original_token)
        vision_enrichment._VISION_IMAGE_SUMMARY_CACHE.clear()

    assert downloaded == [raw_url]
    assert media.images[0].url == proxy_url
    assert "学习包" in media.images[0].summary
    assert any("课堂协作" in line for line in lines)
    assert debug["vision_images_used"] == 1
    assert debug["vision_image_summaries_used"] == 1


@pytest.mark.asyncio
async def test_enrich_media_bundle_puts_all_image_summaries_in_prompt_before_display_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_enabled = settings.vision_enabled
    original_key = settings.vision_api_key
    original_model = settings.vision_model
    original_token = settings.yuque_token
    raw_urls = [
        "https://cdn.nlark.com/yuque/0/2026/png/67444281/teacher.png",
        "https://cdn.nlark.com/yuque/0/2026/png/67444281/software.png",
    ]

    async def fake_download_image(src: str, token: str):
        return src.encode("utf-8"), "image/png"

    async def fake_vision_caption(image_bytes: bytes, mime: str, *, user_hint: str) -> str:
        src = image_bytes.decode("utf-8")
        if "teacher" in src:
            return "教师支持图：展示教师专业发展和培训支持。"
        return "学习软件图：展示软件、课程内容和技术支持。"

    monkeypatch.setattr(vision_enrichment, "_download_image", fake_download_image)
    monkeypatch.setattr(vision_enrichment, "_vision_caption", fake_vision_caption)
    vision_enrichment._VISION_IMAGE_SUMMARY_CACHE.clear()
    try:
        object.__setattr__(settings, "vision_enabled", True)
        object.__setattr__(settings, "vision_api_key", "fake-key")
        object.__setattr__(settings, "vision_model", "fake-qwen-vl")
        object.__setattr__(settings, "yuque_token", "fake-yuque-token")

        media, lines, debug = await vision_enrichment.enrich_media_bundle_with_vision(
            ChatMediaBundle(
                images=[
                    MediaItem(
                        url=f"/yuque/asset?t={encode_image_proxy_token(raw_urls[0])}",
                        title="教师支持",
                    ),
                    MediaItem(
                        url=f"/yuque/asset?t={encode_image_proxy_token(raw_urls[1])}",
                        title="学习软件",
                    ),
                ],
                videos=[],
            ),
            question="乐高人工智能课程",
            max_images=1,
            max_videos=0,
        )
    finally:
        object.__setattr__(settings, "vision_enabled", original_enabled)
        object.__setattr__(settings, "vision_api_key", original_key)
        object.__setattr__(settings, "vision_model", original_model)
        object.__setattr__(settings, "yuque_token", original_token)
        vision_enrichment._VISION_IMAGE_SUMMARY_CACHE.clear()

    assert len(media.images) == 1
    assert any("教师专业发展" in line for line in lines)
    assert any("学习软件" in line for line in lines)
    assert debug["vision_images_used"] == 2
    assert debug["vision_image_summaries_used"] == 2
    assert debug["vision_display_images"] == 1


def test_media_match_score_prioritizes_question_intent_over_original_rank() -> None:
    early_score = vision_enrichment._media_match_score(
        question="我想了解老师培训和教师支持怎么做",
        item=MediaItem(url="https://example.com/early.png", title="乐高课程套装"),
        summary="图片展示核心积木套装、传感器和机器人搭建材料。",
        media_kind="image",
        original_rank=0,
    )
    intent_score = vision_enrichment._media_match_score(
        question="我想了解老师培训和教师支持怎么做",
        item=MediaItem(url="https://example.com/teacher.png", title="教师支持方案"),
        summary="图片展示教师专业发展、老师培训、备课资源和课堂支持。",
        media_kind="image",
        original_rank=20,
    )

    assert intent_score > early_score
