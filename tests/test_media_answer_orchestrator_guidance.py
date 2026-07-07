from __future__ import annotations

import pytest

from app.conversation.lead_nudge_policy import LeadNudgePolicy
from app.db.repositories import ChatMessageRow
from app.schemas.chat import SourceItem
from app.service.media_answer_orchestrator import MediaAnswerOrchestrator


class _FakeMCP:
    enabled = True

    async def search(self, query: str):
        return []

    async def get_doc(self, doc_id: str):
        return ""

    async def list_docs(self):
        class _Doc:
            def __init__(self, title: str) -> None:
                self.title = title

        return [_Doc("平台介绍"), _Doc("使用指南"), _Doc("IDEAS-PBL")]


class _FakeMCPNoList:
    enabled = True

    async def search(self, query: str):
        return []

    async def get_doc(self, doc_id: str):
        return ""

    async def list_docs(self):
        return []


class _FakeMCPNodeContent:
    enabled = True

    async def search(self, query: str):
        return []

    async def get_doc(self, doc_id: str):
        if str(doc_id) == "101":
            return "IDEAS-PBL 是一套面向项目式学习的教学支持系统，帮助老师更高效地组织课堂活动和过程评价。"
        return ""

    async def list_docs(self):
        return []


class _FakeGenerator:
    async def generate(self, *, question: str, contexts, sources, visitor_sales: bool = False):
        return "generated"


class _FakeLeadRepo:
    async def has_lead_for_session(self, *, session_id: str) -> bool:
        return False


class _FakeSessionRepo:
    async def list_recent_messages(self, *, session_id: str, limit: int):
        return [ChatMessageRow(role="user", content="你好", created_at="2026-01-01 10:00:00")]


class _FakeSessionRepoTeacherHistory:
    async def list_recent_messages(self, *, session_id: str, limit: int):
        return [
            ChatMessageRow(role="user", content="你好，我是老师", created_at="2026-01-01 10:00:00"),
            ChatMessageRow(role="assistant", content="好的", created_at="2026-01-01 10:00:01"),
        ]


class _FakeSessionRepoRounds:
    async def list_recent_messages(self, *, session_id: str, limit: int):
        return [
            ChatMessageRow(role="user", content="q1", created_at="2026-01-01 10:00:00"),
            ChatMessageRow(role="assistant", content="a1", created_at="2026-01-01 10:00:01"),
            ChatMessageRow(role="user", content="q2", created_at="2026-01-01 10:00:02"),
            ChatMessageRow(role="assistant", content="a2", created_at="2026-01-01 10:00:03"),
            ChatMessageRow(role="user", content="q3", created_at="2026-01-01 10:00:04"),
            ChatMessageRow(role="assistant", content="a3", created_at="2026-01-01 10:00:05"),
            ChatMessageRow(role="user", content="q4", created_at="2026-01-01 10:00:06"),
        ]


@pytest.mark.asyncio
async def test_answer_guides_when_no_docs() -> None:
    orch = MediaAnswerOrchestrator(
        mcp_client=_FakeMCP(),
        generator=_FakeGenerator(),
        lead_policy=LeadNudgePolicy(rounds_threshold=5, stay_seconds_threshold=120),
        lead_capture_repository=_FakeLeadRepo(),
        chat_session_repository=_FakeSessionRepo(),
        max_images=3,
        max_videos=1,
        max_docs=6,
    )

    resp = await orch.answer(question="你们平台有什么", session_id="s1")
    assert resp.debug and resp.debug.get("guidance_triggered") is True
    assert ("您可以先从这些方向里选一个" in resp.answer) or ("你可以直接复制下面任一问题继续聊" in resp.answer)
    assert resp.sources == []
    assert resp.media.images == []
    assert resp.media.videos == []


@pytest.mark.asyncio
async def test_answer_guides_from_prefetched_titles_first() -> None:
    orch = MediaAnswerOrchestrator(
        mcp_client=_FakeMCPNoList(),
        generator=_FakeGenerator(),
        lead_policy=LeadNudgePolicy(rounds_threshold=5, stay_seconds_threshold=120),
        lead_capture_repository=_FakeLeadRepo(),
        chat_session_repository=_FakeSessionRepo(),
        max_images=3,
        max_videos=1,
        max_docs=6,
        prefetched_titles=["目录A", "目录B", "目录C"],
    )
    resp = await orch.answer(question="你好", session_id="s2")
    assert "目录A" in resp.answer
    assert "目录B" in resp.answer


@pytest.mark.asyncio
async def test_answer_guides_to_second_level_when_root_selected() -> None:
    orch = MediaAnswerOrchestrator(
        mcp_client=_FakeMCPNoList(),
        generator=_FakeGenerator(),
        lead_policy=LeadNudgePolicy(rounds_threshold=5, stay_seconds_threshold=120),
        lead_capture_repository=_FakeLeadRepo(),
        chat_session_repository=_FakeSessionRepo(),
        max_images=3,
        max_videos=1,
        max_docs=6,
        prefetched_toc_nodes=[
            {"uuid": "r1", "title": "平台介绍", "level": 1, "parent_uuid": ""},
            {"uuid": "c1", "title": "课程产品矩阵", "level": 2, "parent_uuid": "r1"},
            {"uuid": "c2", "title": "智能招生平台", "level": 2, "parent_uuid": "r1"},
        ],
    )
    resp = await orch.answer(question="我想看平台介绍", session_id="s4")
    assert "围绕《平台介绍》" in resp.answer
    assert "课程产品矩阵" in resp.answer


@pytest.mark.asyncio
async def test_answer_guides_with_brief_when_parent_has_content() -> None:
    orch = MediaAnswerOrchestrator(
        mcp_client=_FakeMCPNodeContent(),
        generator=_FakeGenerator(),
        lead_policy=LeadNudgePolicy(rounds_threshold=5, stay_seconds_threshold=120),
        lead_capture_repository=_FakeLeadRepo(),
        chat_session_repository=_FakeSessionRepo(),
        max_images=3,
        max_videos=1,
        max_docs=6,
        prefetched_toc_nodes=[
            {"uuid": "r1", "title": "优秀案例库", "level": 1, "parent_uuid": "", "doc_id": 101},
            {"uuid": "c1", "title": "IDEAS-PBL", "level": 2, "parent_uuid": "r1"},
            {"uuid": "c2", "title": "课程产品", "level": 2, "parent_uuid": "r1"},
        ],
    )
    resp = await orch.answer(question="我想了解优秀案例库", session_id="s4y")
    assert "已收到，我先给您一个简要说明" in resp.answer
    assert "面向项目式学习" in resp.answer
    assert "IDEAS-PBL" in resp.answer
    assert "课程产品" in resp.answer


@pytest.mark.asyncio
async def test_answer_guides_root_level_when_no_toc_keyword_match() -> None:
    orch = MediaAnswerOrchestrator(
        mcp_client=_FakeMCPNoList(),
        generator=_FakeGenerator(),
        lead_policy=LeadNudgePolicy(rounds_threshold=5, stay_seconds_threshold=120),
        lead_capture_repository=_FakeLeadRepo(),
        chat_session_repository=_FakeSessionRepo(),
        max_images=3,
        max_videos=1,
        max_docs=6,
        prefetched_toc_nodes=[
            {"uuid": "r1", "title": "平台介绍", "level": 1, "parent_uuid": ""},
            {"uuid": "c1", "title": "课程产品矩阵", "level": 2, "parent_uuid": "r1"},
            {"uuid": "l1", "title": "乐高人工智能课程", "level": 3, "parent_uuid": "c1"},
            {"uuid": "r2", "title": "使用指南", "level": 1, "parent_uuid": ""},
        ],
    )
    resp = await orch.answer(question="你好，我是老师", session_id="s4x")
    assert "您可以先从这些方向里选一个" in resp.answer
    assert "- 平台介绍" in resp.answer
    assert "- 使用指南" in resp.answer


@pytest.mark.asyncio
async def test_answer_guides_to_third_level_when_second_selected() -> None:
    orch = MediaAnswerOrchestrator(
        mcp_client=_FakeMCPNoList(),
        generator=_FakeGenerator(),
        lead_policy=LeadNudgePolicy(rounds_threshold=5, stay_seconds_threshold=120),
        lead_capture_repository=_FakeLeadRepo(),
        chat_session_repository=_FakeSessionRepo(),
        max_images=3,
        max_videos=1,
        max_docs=6,
        prefetched_toc_nodes=[
            {"uuid": "r1", "title": "平台介绍", "level": 1, "parent_uuid": ""},
            {"uuid": "c1", "title": "课程产品矩阵", "level": 2, "parent_uuid": "r1"},
            {"uuid": "l1", "title": "乐高人工智能课程", "level": 3, "parent_uuid": "c1"},
        ],
    )
    resp = await orch.answer(question="我想看课程产品矩阵", session_id="s5")
    assert "围绕《课程产品矩阵》" in resp.answer
    assert "乐高人工智能课程" in resp.answer


@pytest.mark.asyncio
async def test_answer_guides_extract_content_when_leaf_selected() -> None:
    orch = MediaAnswerOrchestrator(
        mcp_client=_FakeMCPNoList(),
        generator=_FakeGenerator(),
        lead_policy=LeadNudgePolicy(rounds_threshold=5, stay_seconds_threshold=120),
        lead_capture_repository=_FakeLeadRepo(),
        chat_session_repository=_FakeSessionRepo(),
        max_images=3,
        max_videos=1,
        max_docs=6,
        prefetched_toc_nodes=[
            {"uuid": "r1", "title": "平台介绍", "level": 1, "parent_uuid": ""},
            {"uuid": "c1", "title": "课程产品矩阵", "level": 2, "parent_uuid": "r1"},
            {"uuid": "l1", "title": "乐高人工智能课程", "level": 3, "parent_uuid": "c1"},
        ],
    )
    resp = await orch.answer(question="请提取乐高人工智能课程内容", session_id="s6")
    assert "当前已定位：《乐高人工智能课程》" in resp.answer
    assert "请提取《乐高人工智能课程》的核心内容" in resp.answer


@pytest.mark.asyncio
async def test_answer_guides_with_role_hint_for_teacher() -> None:
    orch = MediaAnswerOrchestrator(
        mcp_client=_FakeMCP(),
        generator=_FakeGenerator(),
        lead_policy=LeadNudgePolicy(rounds_threshold=5, stay_seconds_threshold=120),
        lead_capture_repository=_FakeLeadRepo(),
        chat_session_repository=_FakeSessionRepo(),
        max_images=3,
        max_videos=1,
        max_docs=6,
    )
    resp = await orch.answer(question="我是老师，想了解平台怎么用", session_id="s3")
    assert "老师您好" in resp.answer


@pytest.mark.asyncio
async def test_answer_guides_keeps_teacher_role_from_history() -> None:
    orch = MediaAnswerOrchestrator(
        mcp_client=_FakeMCPNoList(),
        generator=_FakeGenerator(),
        lead_policy=LeadNudgePolicy(rounds_threshold=5, stay_seconds_threshold=120),
        lead_capture_repository=_FakeLeadRepo(),
        chat_session_repository=_FakeSessionRepoTeacherHistory(),
        max_images=3,
        max_videos=1,
        max_docs=6,
        prefetched_toc_nodes=[
            {"uuid": "r1", "title": "案例与社区", "level": 1, "parent_uuid": ""},
            {"uuid": "c1", "title": "优秀案例库", "level": 2, "parent_uuid": "r1"},
        ],
    )
    resp = await orch.answer(question="我想看看案例", session_id="s3-history")
    assert "老师您好" in resp.answer
    assert "你也可以先告诉我你的身份" not in resp.answer


@pytest.mark.asyncio
async def test_guidance_branch_triggers_lead_nudge_after_round_5() -> None:
    orch = MediaAnswerOrchestrator(
        mcp_client=_FakeMCPNoList(),
        generator=_FakeGenerator(),
        lead_policy=LeadNudgePolicy(rounds_threshold=5, stay_seconds_threshold=120),
        lead_capture_repository=_FakeLeadRepo(),
        chat_session_repository=_FakeSessionRepoRounds(),
        max_images=3,
        max_videos=1,
        max_docs=6,
    )
    resp = await orch.answer(question="我想继续了解", session_id="s-nudge")
    assert resp.lead_nudge_triggered is True
    assert resp.debug and resp.debug.get("lead_nudge_reason") == "rounds"
    assert ("申请测试账号" in resp.answer) or ("留下联系方式" in resp.answer)
