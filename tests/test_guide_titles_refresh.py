import asyncio
import time

import pytest

from app.core.config import settings
from app.service.qa_service import QAService


class _Node:
    def __init__(self, title: str) -> None:
        self.title = title


class _Doc:
    def __init__(self, title: str) -> None:
        self.title = title


class _FakeLoader:
    def __init__(self) -> None:
        self.toc_calls = 0

    async def get_book_toc(self, *, book: str):
        self.toc_calls += 1
        return [_Node("平台介绍"), _Node("使用指南"), _Node("案例与社区")]

    async def list_docs(self, *, book: str, offset: int, limit: int):
        return [_Doc("平台介绍"), _Doc("使用指南"), _Doc("案例与社区")]


@pytest.mark.asyncio
async def test_refresh_guide_doc_titles_respects_interval() -> None:
    loader = _FakeLoader()

    svc = QAService.__new__(QAService)
    svc._yuque_loader = loader
    svc._guide_doc_titles = []
    svc._guide_titles_refreshed_at = 0.0
    svc._guide_titles_refresh_lock = asyncio.Lock()

    old_scope = settings.yuque_scope
    old_refresh = settings.chat_v15_guide_refresh_s
    object.__setattr__(settings, "yuque_scope", "owner/repo")
    object.__setattr__(settings, "chat_v15_guide_refresh_s", 300)
    try:
        await svc._refresh_guide_doc_titles_if_stale(force=True)
        assert loader.toc_calls == 1
        assert "案例与社区" in svc._guide_doc_titles

        await svc._refresh_guide_doc_titles_if_stale()
        assert loader.toc_calls == 1

        svc._guide_titles_refreshed_at = time.monotonic() - 301
        await svc._refresh_guide_doc_titles_if_stale()
        assert loader.toc_calls == 2
    finally:
        object.__setattr__(settings, "yuque_scope", old_scope)
        object.__setattr__(settings, "chat_v15_guide_refresh_s", old_refresh)
