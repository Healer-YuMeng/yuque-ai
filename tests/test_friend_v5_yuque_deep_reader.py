from __future__ import annotations

import pytest

from app.data.mcp_client import MCPSearchResult
from app.data.yuque_loader import YuqueDocument, YuqueSearchHit
from app.schemas.chat import ChatMediaBundle, MediaItem
from app.service.friend_v5_yuque_deep_reader import FriendV5YuqueDeepReader


class _FakeMCP:
    enabled = True

    def __init__(self) -> None:
        self.search_calls: list[str] = []
        self.get_doc_calls: list[str] = []

    async def search(self, query: str):
        self.search_calls.append(query)
        return [
            MCPSearchResult(
                doc_id="101",
                title="乐高人工智能课程介绍",
                url="https://www.yuque.com/example/repo/lego-ai",
                snippet="课程介绍摘要",
            )
        ]

    async def get_doc(self, doc_id: str) -> str:
        self.get_doc_calls.append(doc_id)
        return (
            "# 乐高人工智能课程介绍\n"
            "这篇文档介绍课程目标、适合年级和课堂流程。\n"
            "![课堂搭建图](https://cdn.nlark.com/yuque/0/2026/png/123456/lego.png)\n"
            "[课程演示视频](https://example.com/lego-demo.mp4)"
        )


class _FakeMCPWithManyMedia:
    enabled = True

    def __init__(self) -> None:
        self.search_calls: list[str] = []
        self.get_doc_calls: list[str] = []

    async def search(self, query: str):
        self.search_calls.append(query)
        return []

    async def get_doc(self, doc_id: str) -> str:
        self.get_doc_calls.append(doc_id)
        return (
            "# 苹果STEAM课程\n"
            "这篇文档介绍课程目标、课堂作品和教学效果。\n"
            "![二维码](https://cdn.nlark.com/yuque/0/2026/png/123456/qr.png)\n"
            "![课堂作品](https://cdn.nlark.com/yuque/0/2026/png/123456/work.png)\n"
            "![课程海报](https://cdn.nlark.com/yuque/0/2026/png/123456/poster.png)\n"
            "![教师培训](https://cdn.nlark.com/yuque/0/2026/png/123456/training.png)\n"
            "[课程演示视频](https://example.com/apple-demo.mp4)"
        )


class _FakeMCPWithFiveUsefulImages:
    enabled = True

    async def search(self, query: str):  # noqa: ANN001
        return []

    async def get_doc(self, doc_id: str) -> str:
        return (
            "# 乐高人工智能课程\n"
            "这篇文档介绍课程目标、教师支持、学习软件、课堂活动和教学效果。\n"
            "![课程目标](https://cdn.nlark.com/yuque/0/2026/png/123456/goal.png)\n"
            "![教师支持](https://cdn.nlark.com/yuque/0/2026/png/123456/teacher.png)\n"
            "![学习软件](https://cdn.nlark.com/yuque/0/2026/png/123456/software.png)\n"
            "![课堂活动](https://cdn.nlark.com/yuque/0/2026/png/123456/classroom.png)\n"
            "![教学效果](https://cdn.nlark.com/yuque/0/2026/png/123456/effect.png)"
        )


class _FakeMCPWithManyUsefulImages:
    enabled = True

    async def search(self, query: str):  # noqa: ANN001
        return []

    async def get_doc(self, doc_id: str) -> str:
        images = "\n".join(
            f"![课堂图{i}](https://cdn.nlark.com/yuque/0/2026/png/123456/classroom-{i}.png)"
            for i in range(25)
        )
        return "# 使用指南\n这篇文档包含很多课堂操作截图。\n" + images


class _DisabledMCP:
    enabled = False


class _FakeYuqueLoader:
    def __init__(self) -> None:
        self.search_calls: list[str] = []
        self.get_doc_calls: list[tuple[str | int, str]] = []

    async def search_docs(self, query: str):
        self.search_calls.append(query)
        return [
            YuqueSearchHit(
                title="学校 AI 场景定制指南",
                url="https://www.yuque.com/example/repo/school-ai",
                summary="指南摘要",
                book_id=7,
                doc_id=202,
                slug="school-ai",
            )
        ]

    async def get_doc(self, *, book: str | int, id_or_slug: str):
        self.get_doc_calls.append((book, id_or_slug))
        return YuqueDocument(
            doc_id="202",
            title="学校 AI 场景定制指南",
            url="https://www.yuque.com/example/repo/school-ai",
            body="正文说明实验室、校本课程和教师培训。",
        )


async def _fake_vision_enricher(
    media: ChatMediaBundle,
    *,
    question: str,
    max_images: int,
    max_videos: int,
):
    assert max_images == 2
    assert max_videos == 1
    images = [
        MediaItem(
            url=item.url,
            title=item.title,
            doc_title=item.doc_title,
            doc_id=item.doc_id,
            summary=f"Qwen识别：{item.title or item.url}",
        )
        for item in media.images
        if "qr" not in item.url
    ][:max_images]
    videos = [
        MediaItem(
            url=item.url,
            title=item.title,
            doc_title=item.doc_title,
            doc_id=item.doc_id,
            summary="Qwen识别：课程演示视频展示课堂作品。",
        )
        for item in media.videos
    ][:max_videos]
    return ChatMediaBundle(images=images, videos=videos), ["【文档多媒体识读摘要】", "Qwen识别：课程演示视频展示课堂作品。"], {
        "vision_images_used": len(images),
        "vision_videos_used": len(videos),
        "vision_model": "fake-qwen-vl",
    }


async def _fake_all_image_vision_enricher(
    media: ChatMediaBundle,
    *,
    question: str,
    max_images: int,
    max_videos: int,
):
    assert len(media.images) == 2
    assert max_images == 2
    assert max_videos == 0
    images = [
        MediaItem(
            url=item.url,
            title=item.title,
            doc_title=item.doc_title,
            doc_id=item.doc_id,
            summary=f"Qwen识别：{item.title}",
        )
        for item in media.images
    ]
    selected = [item for item in images if item.title in {"教师支持", "学习软件"}]
    return ChatMediaBundle(images=selected, videos=[]), [
        "【文档多媒体识读摘要】",
        *[item.summary for item in images],
    ], {
        "vision_images_used": len(images),
        "vision_image_summaries_used": len(images),
        "vision_model": "fake-qwen-vl",
    }


async def _should_not_call_vision_enricher(
    media: ChatMediaBundle,
    *,
    question: str,
    max_images: int,
    max_videos: int,
):
    raise AssertionError("媒体过多时应走快速路径，不应同步调用视觉模型")


@pytest.mark.asyncio
async def test_deep_reader_prefers_mcp_get_doc_and_extracts_media() -> None:
    mcp = _FakeMCP()
    reader = FriendV5YuqueDeepReader(mcp_client=mcp, yuque_loader=None, max_images=2, max_videos=1)

    result = await reader.read(question="帮我总结乐高人工智能课程介绍那篇语雀文档")

    assert result.used is True
    assert mcp.search_calls == ["帮我总结乐高人工智能课程介绍那篇语雀文档"]
    assert mcp.get_doc_calls == ["101"]
    assert "乐高人工智能课程介绍" in result.prompt_block
    assert "课程目标" in result.prompt_block
    assert result.sources[0].source_type == "yuque"
    assert result.sources[0].doc_id == "101"
    assert len(result.media.images) == 1
    assert result.media.images[0].url.startswith("/yuque/asset")
    assert len(result.media.videos) == 1
    assert result.media.videos[0].url == "https://example.com/lego-demo.mp4"


@pytest.mark.asyncio
async def test_deep_reader_reads_toc_node_directly_and_enriches_top_three_media() -> None:
    mcp = _FakeMCPWithManyMedia()
    reader = FriendV5YuqueDeepReader(
        mcp_client=mcp,
        yuque_loader=None,
        max_images=4,
        max_videos=1,
        media_enricher=_fake_vision_enricher,
    )

    result = await reader.read_toc_node(
        node={
            "doc_id": "apple-doc",
            "title": "苹果STEAM课程",
            "url": "https://www.yuque.com/example/repo/apple-steam",
        },
        question="苹果STEAM课程",
    )

    assert result.used is True
    assert mcp.search_calls == []
    assert mcp.get_doc_calls == ["apple-doc"]
    assert result.sources[0].doc_id == "apple-doc"
    assert result.sources[0].title == "苹果STEAM课程"
    assert len(result.media.videos) == 1
    assert len(result.media.images) == 2
    assert len(result.media.images) + len(result.media.videos) == 3
    assert all("qr" not in item.url for item in result.media.images)
    assert "Qwen识别：课程演示视频展示课堂作品" in result.prompt_block
    assert result.debug["vision_images_used"] == 2
    assert result.debug["vision_videos_used"] == 1


@pytest.mark.asyncio
async def test_deep_reader_prefilters_doc_images_before_vision_selection() -> None:
    reader = FriendV5YuqueDeepReader(
        mcp_client=_FakeMCPWithFiveUsefulImages(),
        yuque_loader=None,
        max_images=2,
        max_videos=0,
        media_enricher=_fake_all_image_vision_enricher,
    )

    result = await reader.read_toc_node(
        node={
            "doc_id": "lego-doc",
            "title": "乐高人工智能课程",
            "url": "https://www.yuque.com/example/repo/lego-ai",
        },
        question="教师支持和学习软件",
    )

    assert len(result.media.images) == 2
    assert [item.title for item in result.media.images] == ["教师支持", "学习软件"]
    assert "Qwen识别：课程目标" not in result.prompt_block
    assert "Qwen识别：教学效果" not in result.prompt_block
    assert result.debug["candidate_media_images"] == 5
    assert result.debug["vision_prefilter_images"] == 2
    assert result.debug["vision_images_used"] == 2


@pytest.mark.asyncio
async def test_deep_reader_skips_sync_vision_when_doc_has_too_many_images() -> None:
    reader = FriendV5YuqueDeepReader(
        mcp_client=_FakeMCPWithManyUsefulImages(),
        yuque_loader=None,
        max_images=4,
        max_videos=0,
        media_enricher=_should_not_call_vision_enricher,
    )

    result = await reader.read_toc_node(
        node={
            "doc_id": "guide-doc",
            "title": "使用指南",
            "url": "https://www.yuque.com/example/repo/guide",
        },
        question="使用指南",
    )

    assert result.used is True
    assert result.debug["candidate_media_images"] == 25
    assert result.debug["vision_prefilter_images"] == 2
    assert result.debug["vision_media_skipped"] == "too_many_media_fast_path"
    assert len(result.media.images) == 2


@pytest.mark.asyncio
async def test_deep_reader_falls_back_to_yuque_openapi_when_mcp_disabled() -> None:
    loader = _FakeYuqueLoader()
    reader = FriendV5YuqueDeepReader(
        mcp_client=_DisabledMCP(),
        yuque_loader=loader,
        scope="fallback/repo",
        max_images=2,
        max_videos=1,
    )

    result = await reader.read(question="总结学校 AI 场景定制指南")

    assert result.used is True
    assert loader.search_calls == ["总结学校 AI 场景定制指南"]
    assert loader.get_doc_calls == [(7, "202")]
    assert result.sources[0].title == "学校 AI 场景定制指南"
    assert "教师培训" in result.prompt_block
