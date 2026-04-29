import pytest
from types import SimpleNamespace

from app.data.mcp_client import MCPSearchResult
from app.rag.retriever import Retriever


class FakeEmbedder:
    async def embed_query(self, text: str):
        return [1.0, 0.0]


class FakeVectorStore:
    def __init__(self, hits):
        self._hits = hits

    def search(self, query_embedding, top_k):
        return self._hits[:top_k]


class FakeMCPClient:
    def __init__(self, results):
        self._results = results

    async def search(self, query: str):
        return self._results


@pytest.mark.asyncio
async def test_retriever_prefers_vector_hits() -> None:
    retriever = Retriever(
        vector_store=FakeVectorStore(
            [
                SimpleNamespace(
                    chunk=SimpleNamespace(
                        chunk_id="1",
                        doc_id="doc-1",
                        title="向量文档",
                        url="https://example.com/vector",
                        text="这是向量检索命中的内容",
                        order=0,
                    ),
                    score=0.9,
                )
            ]
        ),
        embedder=FakeEmbedder(),
        mcp_client=FakeMCPClient([]),
        yuque_loader=None,
        top_k=3,
        score_threshold=0.5,
    )

    result = await retriever.retrieve("问题")

    assert result.fallback_used is False
    assert result.sources[0].source_type == "vector"


@pytest.mark.asyncio
async def test_retriever_falls_back_to_mcp() -> None:
    retriever = Retriever(
        vector_store=FakeVectorStore([]),
        embedder=FakeEmbedder(),
        mcp_client=FakeMCPClient(
            [
                MCPSearchResult(
                    doc_id="doc-mcp",
                    title="MCP文档",
                    url="https://example.com/mcp",
                    snippet="来自MCP的实时结果",
                )
            ]
        ),
        yuque_loader=None,
        top_k=3,
        score_threshold=0.5,
    )

    result = await retriever.retrieve("问题")

    assert result.fallback_used is True
    assert result.sources[0].source_type == "mcp"


class FakeYuqueLoader:
    async def search_docs(self, query: str):
        self.last_query = query
        return [
            SimpleNamespace(
                title="语雀文档",
                url="https://example.com/yuque",
                summary="摘要",
                book_id=1,
                doc_id=2,
                slug="doc",
            )
        ]

    async def get_doc(self, *, book, id_or_slug: str):
        return SimpleNamespace(title="语雀文档", url="https://example.com/yuque", body="来自语雀正文")

    async def get_book_toc(self, *, book):
        return [
            SimpleNamespace(
                uuid="root-1",
                type="doc",
                title="最终状态机",
                url="https://example.com/toc/final",
                doc_id=2,
                level=1,
                parent_uuid="",
            ),
            SimpleNamespace(
                uuid="child-1",
                type="doc",
                title="状态流转",
                url="https://example.com/toc/flow",
                doc_id=3,
                level=2,
                parent_uuid="root-1",
            ),
        ]

    async def list_docs(self, *, book, offset=0, limit=50):
        return [
            SimpleNamespace(id=1, slug="doc-a", title="最终状态机", url="https://example.com/doc/a", updated_at=""),
            SimpleNamespace(id=2, slug="doc-b", title="状态流转", url="https://example.com/doc/b", updated_at=""),
        ]

    _scope = "team/book"


@pytest.mark.asyncio
async def test_retriever_uses_direct_yuque_when_no_embedder() -> None:
    loader = FakeYuqueLoader()
    retriever = Retriever(
        vector_store=FakeVectorStore([]),
        embedder=None,
        mcp_client=FakeMCPClient([]),
        yuque_loader=loader,
        top_k=3,
        score_threshold=0.5,
    )

    result = await retriever.retrieve("问题")

    assert result.fallback_used is False
    assert result.sources[0].source_type == "yuque"
    assert loader.last_query == "问题"


@pytest.mark.asyncio
async def test_retriever_extracts_core_phrase_for_search() -> None:
    loader = FakeYuqueLoader()
    retriever = Retriever(
        vector_store=FakeVectorStore([]),
        embedder=None,
        mcp_client=FakeMCPClient([]),
        yuque_loader=loader,
        top_k=3,
        score_threshold=0.5,
    )

    await retriever.retrieve("最终状态机是什么内容")

    assert loader.last_query == "最终状态机"


@pytest.mark.asyncio
async def test_retriever_uses_toc_for_directory_questions() -> None:
    retriever = Retriever(
        vector_store=FakeVectorStore([]),
        embedder=None,
        mcp_client=FakeMCPClient([]),
        yuque_loader=FakeYuqueLoader(),
        top_k=3,
        score_threshold=0.5,
    )

    result = await retriever.retrieve("有什么目录")

    assert result.fallback_used is False
    assert "目录" in result.contexts[0]
    assert result.sources[0].title == "知识库目录"


@pytest.mark.asyncio
async def test_retriever_uses_toc_for_subdirectory_questions() -> None:
    retriever = Retriever(
        vector_store=FakeVectorStore([]),
        embedder=None,
        mcp_client=FakeMCPClient([]),
        yuque_loader=FakeYuqueLoader(),
        top_k=3,
        score_threshold=0.5,
    )

    result = await retriever.retrieve("有子目录吗？")

    assert result.fallback_used is False
    assert "目录" in result.contexts[0]
    assert result.sources[0].title == "知识库目录"
    assert "最终状态机" in result.contexts[0]
    assert "状态流转" in result.contexts[0]


@pytest.mark.asyncio
async def test_retriever_uses_docs_list_for_doc_list_questions() -> None:
    retriever = Retriever(
        vector_store=FakeVectorStore([]),
        embedder=None,
        mcp_client=FakeMCPClient([]),
        yuque_loader=FakeYuqueLoader(),
        top_k=3,
        score_threshold=0.5,
    )

    result = await retriever.retrieve("现在有哪些语雀文档")

    assert result.fallback_used is False
    assert "文档列表" in result.contexts[0]
    assert "最终状态机" in result.contexts[0]
    assert result.sources[0].title == "知识库文档列表"


@pytest.mark.asyncio
async def test_retriever_uses_docs_title_match_for_content_questions() -> None:
    class SearchMissYuqueLoader(FakeYuqueLoader):
        async def search_docs(self, query: str):
            self.last_query = query
            return []

        async def list_docs(self, *, book, offset=0, limit=50):
            return [
                SimpleNamespace(id=101, slug="step-3", title="成大事三步法", url="https://example.com/doc/step3", updated_at="")
            ]

        async def get_doc(self, *, book, id_or_slug: str):
            return SimpleNamespace(title="成大事三步法", url="https://example.com/doc/step3", body="第一步...第二步...第三步...")

    retriever = Retriever(
        vector_store=FakeVectorStore([]),
        embedder=None,
        mcp_client=FakeMCPClient([]),
        yuque_loader=SearchMissYuqueLoader(),
        top_k=3,
        score_threshold=0.5,
    )

    result = await retriever.retrieve("成大事三步法是什么")

    assert result.fallback_used is False
    assert result.sources[0].title == "成大事三步法"
    assert "第一步" in result.contexts[0]

