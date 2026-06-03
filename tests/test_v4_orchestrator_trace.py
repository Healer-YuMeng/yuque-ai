from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Dict, List, Sequence

import pytest

from app.conversation.catalog_state_machine import CatalogDialogState
from app.conversation.toc_catalog import CatalogNode
from app.conversation.turn_trace import TurnTraceBuilder
from app.data.mcp_client import MCPSearchResult
from app.db.repositories import ChatMessageRow
from app.service.sales_dialog_orchestrator_v4 import SalesDialogOrchestratorV4


class _FakeMCP:
    def __init__(self) -> None:
        self.search_calls = 0

    async def search(self, query: str) -> List[MCPSearchResult]:
        self.search_calls += 1
        return [MCPSearchResult(doc_id="99", title="跨学科项目式学习", url="", snippet="snippet")]

    async def get_doc(self, doc_id: str) -> str:
        return "正文内容" * 50


class _FakeGenerator:
    async def stream_generate(self, **kwargs: Any) -> AsyncIterator[str]:
        yield "回答"


class _FakeProfileRepo:
    async def get_profile(self, *, session_id: str):
        return None

    async def get_catalog_state(self, *, session_id: str):
        return CatalogDialogState(path_titles=["案例与社区"], dialog_level=2)

    async def save_catalog_state(self, *, session_id: str, state: CatalogDialogState) -> None:
        return None

    async def upsert_profile(self, **kwargs: Any) -> None:
        return None

    async def touch_focus_docs(self, *, session_id: str, doc_ids: Sequence[str]) -> None:
        return None


class _FakeLead:
    async def ingest_user_turn(self, **kwargs: Any):
        from app.conversation.v4_lead_outreach import V4LeadTurnResult

        return V4LeadTurnResult(lead_saved=False, contact_detected=False, interests_patch={})

    async def evaluate_end_of_turn(self, **kwargs: Any):
        from app.conversation.lead_nudge_policy import LeadNudgeDecision
        from app.conversation.v4_lead_outreach import V4LeadEndResult

        return V4LeadEndResult(
            append_text="",
            nudge=LeadNudgeDecision(triggered=False, reason=""),
            trial_apply_available=False,
        )


@pytest.mark.asyncio
async def test_v4_retrieve_records_turn_trace() -> None:
    fake_mcp = _FakeMCP()
    orch = SalesDialogOrchestratorV4(
        mcp_client=fake_mcp,
        generator=_FakeGenerator(),
        profile_repo=_FakeProfileRepo(),
        toc_nodes=[
            {
                "title": "跨学科项目式学习",
                "doc_id": 88,
                "uuid": "u1",
                "level": 2,
                "parent_uuid": "",
                "node_type": "DOC",
            }
        ],
        lead_outreach=_FakeLead(),
    )
    node = CatalogNode(
        uuid="u1",
        title="跨学科项目式学习",
        level=2,
        parent_uuid="",
        node_type="DOC",
        url=None,
        doc_id=88,
        path_titles=["案例与社区", "跨学科项目式学习"],
    )
    trace = TurnTraceBuilder(pipeline="v4_content", catalog_path=node.path_titles, dialog_level=2)
    await orch._retrieve_for_nodes([node], catalog_path=node.path_titles, trace=trace, primary_title=node.title)
    built = trace.build()
    assert any(c.tool == "yuque_search" for c in built.mcp_calls)
    assert any(c.tool == "yuque_get_doc" for c in built.mcp_calls)
    assert built.documents


@pytest.mark.asyncio
async def test_v4_retrieve_prefers_cached_payload_for_follow_up() -> None:
    fake_mcp = _FakeMCP()
    orch = SalesDialogOrchestratorV4(
        mcp_client=fake_mcp,
        generator=_FakeGenerator(),
        profile_repo=_FakeProfileRepo(),
        toc_nodes=[
            {
                "title": "跨学科项目式学习",
                "doc_id": 88,
                "uuid": "u1",
                "level": 2,
                "parent_uuid": "",
                "node_type": "DOC",
            }
        ],
        lead_outreach=_FakeLead(),
    )
    node = CatalogNode(
        uuid="u1",
        title="跨学科项目式学习",
        level=2,
        parent_uuid="",
        node_type="DOC",
        url=None,
        doc_id=88,
        path_titles=["案例与社区", "跨学科项目式学习"],
    )
    await orch._retrieve_for_nodes(
        [node],
        catalog_path=node.path_titles,
        trace=None,
        primary_title=node.title,
        session_id="s1",
        cache_scope=node.uuid,
        prefer_cached=False,
    )
    first_search_calls = fake_mcp.search_calls
    doc_ctx, sources, debug = await orch._retrieve_for_nodes(
        [node],
        catalog_path=node.path_titles,
        trace=None,
        primary_title=node.title,
        session_id="s1",
        cache_scope=node.uuid,
        prefer_cached=True,
    )
    assert doc_ctx
    assert sources
    assert debug["retrieval_cache_hit"] is True
    assert fake_mcp.search_calls == first_search_calls
