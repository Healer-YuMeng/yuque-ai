from __future__ import annotations

from app.service.qa_service import (
    _generating_stage_detail,
    _retrieving_stage_detail,
    _sse_stage,
)


def test_sse_stage_shape() -> None:
    ev = _sse_stage("retrieving", "测试中", mode="rag")
    assert ev["event"] == "stage"
    assert ev["data"]["stage"] == "retrieving"
    assert ev["data"]["detail"] == "测试中"
    assert ev["data"]["mode"] == "rag"


def test_retrieving_detail_with_skill_and_scope() -> None:
    s = _retrieving_stage_detail(runtime_label="RAG 向量模式", skill_id="smart-search", scope="a/b")
    assert "RAG 向量模式" in s
    assert "smart-search" in s
    assert "a/b" in s


def test_generating_detail_by_retrieval_mode() -> None:
    assert "助手" in _generating_stage_detail({"retrieval_mode": "scope_help_direct"})
    assert "文档列表" in _generating_stage_detail({"retrieval_mode": "stale_detector"})
    assert "MCP" in _generating_stage_detail({"retrieval_mode": "mcp_fallback"})
    assert "大模型" in _generating_stage_detail({"retrieval_mode": "yuque_or_vector"})
