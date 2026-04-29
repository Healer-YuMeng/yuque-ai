from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.data.yuque_loader import YuqueLoaderError
from app.rag.retriever import Retriever, RetrievalResult
from app.schemas.chat import SourceItem


class FakeVectorStore:
    def search(self, query_embedding, top_k):
        return []


class FakeMCPClient:
    # stale-detector 不会用到 mcp_client，这里仅占位。
    enabled = False
    repo_id = ""


class FakeYuqueLoader:
    def __init__(self) -> None:
        self._scope = "fenyuansaki/smocxp"

    async def list_docs(self, *, book, offset=0, limit=60):
        if not book:
            raise YuqueLoaderError("missing book")
        return [
            SimpleNamespace(id=1, title="DocA", url="https://example.com/doc/a", updated_at="2024-01-01", slug="doc-a"),
            SimpleNamespace(id=2, title="DocB", url="https://example.com/doc/b", updated_at="2023-01-01", slug="doc-b"),
        ]

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_stale_detector_builds_context_from_updated_at() -> None:
    retriever = Retriever(
        vector_store=FakeVectorStore(),
        embedder=None,
        mcp_client=FakeMCPClient(),
        yuque_loader=FakeYuqueLoader(),
        top_k=4,
        score_threshold=0.35,
    )

    result = await retriever.retrieve("任意问题", skill_id="stale-detector")
    assert isinstance(result, RetrievalResult)
    assert result.contexts
    assert "updated_at" in result.contexts[0] or "2024-01-01" in result.contexts[0]
    assert result.sources
    assert result.debug.get("skill_id") == "stale-detector"

