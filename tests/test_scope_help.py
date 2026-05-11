from __future__ import annotations

from app.rag.retriever import RetrievalResult
from app.rag.scope_help import (
    SCOPE_HELP_CONTEXT,
    apply_scope_help_if_needed,
    contexts_effectively_empty,
    direct_scope_help_retrieval,
    is_assistant_scope_meta_question,
)
from app.schemas.chat import SourceItem


def test_is_assistant_scope_meta_question_positive() -> None:
    assert is_assistant_scope_meta_question("你可以回答知识库以外的问题吗？")
    assert is_assistant_scope_meta_question("能否回答语雀以外的内容")
    assert is_assistant_scope_meta_question("你可以回答 知识库以外 的问题吗")
    assert is_assistant_scope_meta_question("你可以回答其他问题吗？")
    assert is_assistant_scope_meta_question("能否解答别的问题")
    assert is_assistant_scope_meta_question("你可以回答哪些问题？")
    assert is_assistant_scope_meta_question("你有什么功能")
    assert is_assistant_scope_meta_question("你有哪些功能？")


def test_is_assistant_scope_meta_question_negative() -> None:
    assert not is_assistant_scope_meta_question("")
    assert not is_assistant_scope_meta_question("研究生综合服务平台登录流程")
    assert not is_assistant_scope_meta_question("知识库以外的退款多久到账")
    assert not is_assistant_scope_meta_question("什么是 RAG")
    assert not is_assistant_scope_meta_question("你可以回答其他部门的流程吗")
    assert not is_assistant_scope_meta_question("你可以回答退款需要准备哪些问题")


def test_apply_scope_help_when_empty_and_meta() -> None:
    empty = RetrievalResult(
        contexts=[],
        sources=[],
        fallback_used=False,
        debug={"retrieval_mode": "mcp_fallback", "mcp_route": "search_empty"},
    )
    out = apply_scope_help_if_needed(empty, "你可以回答其他问题吗？")
    assert out.debug.get("scope_help_injected") is True
    assert len(out.contexts) == 1
    assert SCOPE_HELP_CONTEXT in out.contexts[0]
    assert len(out.sources) == 1
    assert out.sources[0].title == "助手能力与范围（系统说明）"


def test_apply_scope_help_skips_when_has_context() -> None:
    r = RetrievalResult(
        contexts=["已有正文"],
        sources=[SourceItem(title="某文档", url="http://x", source_type="yuque", snippet="x")],
        fallback_used=False,
        debug={},
    )
    out = apply_scope_help_if_needed(r, "你可以回答知识库以外的问题吗？")
    assert out is r
    assert "scope_help_injected" not in out.debug


def test_direct_scope_help_retrieval() -> None:
    r = direct_scope_help_retrieval()
    assert r.debug.get("retrieval_mode") == "scope_help_direct"
    assert r.debug.get("assistant_meta_route") == "rule"
    assert r.debug.get("scope_help_bypass_retrieval") is True
    assert SCOPE_HELP_CONTEXT in r.contexts[0]
    assert "可提供的能力概览" in r.contexts[0]
    assert r.sources[0].title == "助手能力与范围（系统说明）"


def test_contexts_effectively_empty() -> None:
    assert contexts_effectively_empty([])
    assert contexts_effectively_empty(["", "  "])
    assert not contexts_effectively_empty(["a"])
