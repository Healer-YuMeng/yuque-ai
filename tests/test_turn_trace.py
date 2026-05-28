from __future__ import annotations

from app.conversation.turn_trace import TurnTraceBuilder, empty_guide_trace
from app.core.config import settings


def test_turn_trace_builder_records_mcp_and_docs() -> None:
    trace = TurnTraceBuilder(pipeline="v4_content", catalog_path=["A", "B"], dialog_level=2)
    trace.record_search(query="A B 标题", hit_count=2)
    trace.record_get_doc(doc_id="1", title="主文档", body_chars=1200)
    trace.add_document(doc_id="1", title="主文档", role="primary", snippet="摘录")
    built = trace.build()
    assert built.pipeline == "v4_content"
    assert len(built.mcp_calls) == 2
    assert built.documents[0].role == "primary"


def test_empty_guide_trace_has_empty_lists() -> None:
    dbg = empty_guide_trace(pipeline="v4_guide", catalog_path=[], dialog_level=0)
    assert dbg.get("turn_trace", {}).get("mcp_calls") == []
    assert dbg.get("turn_trace", {}).get("skills") == []


def test_attach_debug_respects_expose_flag() -> None:
    old = settings.expose_turn_trace
    try:
        object.__setattr__(settings, "expose_turn_trace", False)
        trace = TurnTraceBuilder(pipeline="v4_content", dialog_level=2)
        trace.record_search(query="q", hit_count=1)
        assert "turn_trace" not in trace.attach_debug({"mode": "v4_content"})

        object.__setattr__(settings, "expose_turn_trace", True)
        out = trace.attach_debug({"mode": "v4_content"})
        assert "turn_trace" in out
        assert out["turn_trace"]["mcp_calls"]
    finally:
        object.__setattr__(settings, "expose_turn_trace", old)
