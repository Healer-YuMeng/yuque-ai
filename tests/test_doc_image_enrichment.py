from __future__ import annotations

from dataclasses import replace
from unittest.mock import AsyncMock

import pytest

import app.rag.doc_image_enrichment as doc_image_enrichment
from app.data.yuque_loader import YuqueDocument
from app.rag.doc_image_enrichment import (
    _bodies_from_context_chunks,
    _parse_yuque_doc_location,
    _source_fetch_keys,
    enrich_retrieval_with_doc_images,
)
from app.rag.retriever import RetrievalResult
from app.schemas.chat import SourceItem


def test_parse_yuque_doc_location() -> None:
    loc = _parse_yuque_doc_location("https://www.yuque.com/myorg/mybook/abc-slug")
    assert loc == ("myorg/mybook", "abc-slug")


def test_parse_yuque_doc_location_none_for_short_path() -> None:
    assert _parse_yuque_doc_location("https://www.yuque.com/a/b") is None


def test_source_fetch_keys_prefers_doc_id() -> None:
    s = SourceItem(title="t", url=None, source_type="vector", doc_id="12345")
    assert _source_fetch_keys(s, "org/repo") == ("org/repo", "12345")


def test_source_fetch_keys_from_url() -> None:
    s = SourceItem(
        title="t",
        url="https://www.yuque.com/org/repo/doc-page",
        source_type="yuque",
    )
    assert _source_fetch_keys(s, "org/repo") == ("org/repo", "doc-page")


def test_bodies_from_context_chunks_requires_aligned_lengths() -> None:
    s = SourceItem(title="a", url="https://www.yuque.com/o/r/d", source_type="yuque", doc_id="1")
    r = RetrievalResult(
        contexts=["x", "y"],
        sources=[s],
        fallback_used=False,
        debug={},
    )
    assert _bodies_from_context_chunks(r) == []


def test_bodies_from_context_chunks_multi_source_picks_best_title() -> None:
    """多篇命中时只从标题与问题最相关的一篇抽图，避免串图。"""
    url_a = "https://cdn.nlark.com/yuque/0/2024/jpeg/a.jpg"
    url_b = "https://cdn.nlark.com/yuque/0/2024/jpeg/b.jpg"
    r = RetrievalResult(
        contexts=[
            f"文档标题：别的课程\n\n![]({url_a})",
            f"文档标题：乐高人工智能课程\n\n![]({url_b})",
        ],
        sources=[
            SourceItem(title="别的课程", url="https://www.yuque.com/o/r/d1", source_type="mcp", doc_id="1"),
            SourceItem(title="乐高人工智能课程", url="https://www.yuque.com/o/r/d2", source_type="mcp", doc_id="2"),
        ],
        fallback_used=True,
        debug={},
    )
    bodies = _bodies_from_context_chunks(r, question="乐高人工智能课程是否有图片可以给我看看")
    assert len(bodies) == 1
    assert bodies[0][0] == "乐高人工智能课程"
    assert url_b in bodies[0][2][0].src
    assert url_a not in bodies[0][2][0].src


def test_bodies_from_context_chunks_multi_source_no_overlap_skips() -> None:
    r = RetrievalResult(
        contexts=["![](https://cdn.nlark.com/yuque/0/2024/jpeg/x.jpg)", "![](https://cdn.nlark.com/yuque/0/2024/jpeg/y.jpg)"],
        sources=[
            SourceItem(title="文档甲", url="https://www.yuque.com/o/r/a", source_type="mcp", doc_id="1"),
            SourceItem(title="文档乙", url="https://www.yuque.com/o/r/b", source_type="mcp", doc_id="2"),
        ],
        fallback_used=True,
        debug={},
    )
    assert _bodies_from_context_chunks(r, question="完全无关的提问词") == []


@pytest.mark.asyncio
async def test_enrich_markdown_only_without_vision(monkeypatch: pytest.MonkeyPatch) -> None:
    """默认仅从检索片段抽图，不 get_doc 全文；片段中含 Markdown 图链即可追加代理块。"""

    class FakeLoader:
        scope = "org/book"
        get_doc_calls = 0

        async def get_doc(self, *, book: str, id_or_slug: str) -> YuqueDocument:
            self.get_doc_calls += 1
            _ = book, id_or_slug
            body = "说明\n![图](https://cdn.nlark.com/yuque/0/2024/jpeg/demo.jpg)\n"
            return YuqueDocument(
                doc_id="99",
                title="含图文档",
                url="https://www.yuque.com/org/book/doc-slug",
                body=body,
            )

    base = doc_image_enrichment.settings
    monkeypatch.setattr(
        doc_image_enrichment,
        "settings",
        replace(
            base,
            vision_enabled=False,
            doc_images_markdown_in_context=True,
            doc_images_full_document_fallback=False,
            yuque_token="dummy-token",
            yuque_scope="org/book",
            vision_max_images=4,
        ),
    )

    chunk = "检索片段\n![图](https://cdn.nlark.com/yuque/0/2024/jpeg/demo.jpg)\n"
    retrieval = RetrievalResult(
        contexts=[chunk],
        sources=[
            SourceItem(
                title="含图文档",
                url="https://www.yuque.com/org/book/doc-slug",
                source_type="yuque",
                doc_id="99",
            ),
        ],
        fallback_used=False,
        debug={"retrieval_mode": "vector"},
    )
    loader = FakeLoader()
    out = await enrich_retrieval_with_doc_images(
        retrieval=retrieval,
        question="问题",
        loader=loader,  # type: ignore[arg-type]
    )
    assert loader.get_doc_calls == 0
    assert len(out.contexts) == 2
    assert "【文档插图" in out.contexts[-1]
    assert "/yuque/asset?t=" in out.contexts[-1]
    assert out.debug.get("doc_images_markdown_only") is True
    assert out.debug.get("doc_image_markdown_count") == 1
    assert out.debug.get("doc_image_refs_source") == "context_chunks"


@pytest.mark.asyncio
async def test_enrich_full_document_fallback_when_context_has_no_image(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLoader:
        scope = "org/book"
        get_doc_calls = 0

        async def get_doc(self, *, book: str, id_or_slug: str) -> YuqueDocument:
            self.get_doc_calls += 1
            _ = book, id_or_slug
            body = "全文\n![](https://cdn.nlark.com/yuque/0/2024/jpeg/fb.jpg)\n"
            return YuqueDocument(
                doc_id="1",
                title="T",
                url="https://www.yuque.com/org/book/d",
                body=body,
            )

    base = doc_image_enrichment.settings
    monkeypatch.setattr(
        doc_image_enrichment,
        "settings",
        replace(
            base,
            vision_enabled=False,
            doc_images_markdown_in_context=True,
            doc_images_full_document_fallback=True,
            yuque_token="t",
            yuque_scope="org/book",
            vision_max_images=4,
        ),
    )
    retrieval = RetrievalResult(
        contexts=["纯文字检索片段，无图"],
        sources=[
            SourceItem(title="T", url="https://www.yuque.com/org/book/d", source_type="yuque", doc_id="1"),
        ],
        fallback_used=False,
        debug={"retrieval_mode": "vector"},
    )
    loader = FakeLoader()
    out = await enrich_retrieval_with_doc_images(retrieval=retrieval, question="q", loader=loader)  # type: ignore[arg-type]
    assert loader.get_doc_calls == 1
    assert "/yuque/asset?t=" in out.contexts[-1]
    assert out.debug.get("doc_image_refs_source") == "full_document"


@pytest.mark.asyncio
async def test_enrich_markdown_only_respects_off_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLoader:
        scope = "org/book"
        get_doc = AsyncMock()

    base = doc_image_enrichment.settings
    monkeypatch.setattr(
        doc_image_enrichment,
        "settings",
        replace(
            base,
            vision_enabled=False,
            doc_images_markdown_in_context=False,
            yuque_token="t",
            yuque_scope="org/book",
        ),
    )
    retrieval = RetrievalResult(
        contexts=["x"],
        sources=[
            SourceItem(title="T", url="https://www.yuque.com/org/book/d", source_type="yuque", doc_id="1"),
        ],
        fallback_used=False,
        debug={"retrieval_mode": "vector"},
    )
    out = await enrich_retrieval_with_doc_images(
        retrieval=retrieval,
        question="q",
        loader=FakeLoader(),  # type: ignore[arg-type]
    )
    assert out.contexts == retrieval.contexts
    FakeLoader.get_doc.assert_not_called()
