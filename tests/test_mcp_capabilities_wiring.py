"""YuqueMCPClient.read_tools 与 capabilities 清单中的只读工具一致；写工具不在 read_tools 内。"""

from __future__ import annotations

from app.data.mcp_client import YuqueMCPClient

_READONLY_CANONICAL = frozenset(
    {
        "yuque_get_user",
        "yuque_list_books",
        "yuque_get_book",
        "yuque_list_docs",
        "yuque_get_doc",
        "yuque_get_toc",
        "yuque_search",
        "yuque_list_notes",
        "yuque_get_note",
    }
)

_WRITE_ONLY = frozenset(
    {
        "yuque_create_book",
        "yuque_update_book",
        "yuque_create_doc",
        "yuque_update_doc",
        "yuque_update_toc",
        "yuque_create_note",
        "yuque_update_note",
    }
)


def test_read_tools_matches_canonical_when_using_yuque_prefixed_tools() -> None:
    client = YuqueMCPClient(
        command="npx",
        args="",
        repo_id="login/repo",
        search_tool="yuque_search",
        get_doc_tool="yuque_get_doc",
    )
    assert frozenset(client.read_tools) == _READONLY_CANONICAL


def test_read_tools_respects_custom_search_and_get_doc_names() -> None:
    client = YuqueMCPClient(
        command="npx",
        args="",
        repo_id="login/repo",
        search_tool="search",
        get_doc_tool="get_doc",
    )
    tools = frozenset(client.read_tools)
    assert "search" in tools and "get_doc" in tools
    assert "yuque_search" not in tools and "yuque_get_doc" not in tools
    assert _WRITE_ONLY.isdisjoint(tools)
