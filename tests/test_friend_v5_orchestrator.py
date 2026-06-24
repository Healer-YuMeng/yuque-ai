from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from app.conversation.profile_extractor import ProfileUpdate
from app.rag.friend_v5_generator import FriendV5StreamEvent
from app.schemas.chat import ChatMediaBundle, MediaItem
from app.schemas.chat_v5 import FriendV5SourceItem
from app.core.config import settings
from app.db.repositories import ChatMessageRow
from app.service.qa_service import QAService
from app.service.friend_dialog_orchestrator_v5 import (
    FriendDialogOrchestratorV5,
    _CASE_KB_FALLBACK_ANSWER,
    _public_yuque_share_url_for_focus,
)
from app.service.friend_v5_tags import (
    case_tag_for_scene,
    explore_product_tag_for_title,
    guide_tag_for_scene,
    trial_tag_for_scene,
)
from app.service.friend_v5_yuque_deep_reader import FriendV5YuqueDeepReadResult

YUQUE_SHARED_AI_COURSE_URL = (
    "https://www.yuque.com/suesun-yb1bi/sspenu/sbdx665n47rz9rt5?singleDoc#%20《人工智能通识课程》"
)
YUQUE_SHARED_PBL_GUIDE_URL = (
    "https://www.yuque.com/suesun-yb1bi/sspenu/dl4rxzdb0ahgq42n?singleDoc#%20《跨学科项目式学习》"
)
YUQUE_SHARED_SMART_ENROLLMENT_GUIDE_URL = (
    "https://www.yuque.com/suesun-yb1bi/sspenu/pmg3pix4w4e6g1zd?singleDoc#%20《智能招生》"
)
YUQUE_SHARED_AI_CASE_URL = (
    "https://www.yuque.com/suesun-yb1bi/sspenu/pynfez9lydaxq7gg?singleDoc#%20《人工智能通识课程》"
)
YUQUE_SHARED_PBL_CASE_URL = (
    "https://www.yuque.com/suesun-yb1bi/sspenu/ztzk0v4ggl934d86?singleDoc#%20《跨学科项目式学习》"
)
YUQUE_SHARED_CERTIFICATION_CASE_URL = (
    "https://www.yuque.com/suesun-yb1bi/sspenu/kfuc54vihosyzlvo?singleDoc#%20《相关赛事及认证》"
)


@dataclass
class _FakeProfile:
    display_name: str = ""
    org_name: str = ""
    visitor_type: str = ""
    interests: dict[str, Any] | None = None


class _FakeProfileRepo:
    def __init__(self) -> None:
        self.profile = _FakeProfile()
        self.upserts: list[dict[str, Any]] = []

    async def get_profile(self, *, session_id: str):
        return self.profile

    async def upsert_profile(self, **kwargs):
        self.upserts.append(kwargs)
        self.profile = _FakeProfile(
            display_name=kwargs.get("display_name") or self.profile.display_name,
            org_name=kwargs.get("org_name") or self.profile.org_name,
            visitor_type=kwargs.get("visitor_type") or self.profile.visitor_type,
            interests=kwargs.get("interests") or self.profile.interests,
        )


class _FakeProfileExtractor:
    async def extract_update(self, **kwargs):  # noqa: ANN003
        return ProfileUpdate(
            display_name="赵老师",
            org_name="第一中学",
            interests={"topics": ["人工智能通识教育"]},
        )


class _FakeGenerator:
    async def stream(self, *, system_prompt: str, user_prompt: str, enable_search: bool = False):
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.enable_search = enable_search
        yield FriendV5StreamEvent.token("我是小为，先帮你把重点拎出来。")
        yield FriendV5StreamEvent.token(
            "[SOURCES]\n"
            "https://www.youweiai.com/web\n"
            "[/SOURCES]\n"
            "[TAGS]想看课程例子？\n想了解适合年级？\n想看看落地方式？[END_TAGS]"
        )


class _FakeGeneratorWithWebSources:
    async def stream(self, *, system_prompt: str, user_prompt: str, enable_search: bool = False):
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.enable_search = enable_search
        yield FriendV5StreamEvent.web_sources(
            [
                FriendV5SourceItem(
                    source_type="web",
                    title="真实联网来源",
                    url="https://www.youweiai.com/news",
                    snippet="联网搜索摘要",
                    index=1,
                )
            ]
        )
        yield FriendV5StreamEvent.token("我是小为，先说结论。")
        yield FriendV5StreamEvent.token(
            "[SOURCES]\n"
            "example.com/page1 https://example.com/page2 [/SOURCES]\n"
            "[TAGS]想看课程例子？\n想了解适合年级？\n想看看落地方式？[END_TAGS]"
        )


class _FakeYuqueSearch:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def search_docs(self, *, query: str, limit: int):
        self.calls.append(query)
        return [
            FriendV5SourceItem(
                source_type="yuque",
                title="乐高人工智能课程介绍",
                url="https://www.yuque.com/example/lego-ai",
                doc_id="101",
            )
        ]


class _FakeDeepReader:
    def __init__(self, *, used: bool = True) -> None:
        self.calls: list[str] = []
        self.node_calls: list[dict[str, Any]] = []
        self._used = used

    async def read(self, *, question: str) -> FriendV5YuqueDeepReadResult:
        self.calls.append(question)
        return self._result()

    async def read_toc_node(self, *, node: dict[str, Any], question: str) -> FriendV5YuqueDeepReadResult:
        self.node_calls.append({"node": node, "question": question})
        return self._result()

    def _result(self) -> FriendV5YuqueDeepReadResult:
        if not self._used:
            return FriendV5YuqueDeepReadResult(debug={"mode": "yuque_get_doc_by_toc", "empty_body": True})
        return FriendV5YuqueDeepReadResult(
            used=True,
            prompt_block="【语雀文档深读】\n标题：乐高人工智能课程介绍\n正文摘录：课程目标、课堂流程、作品展示。",
            sources=[
                FriendV5SourceItem(
                    source_type="yuque",
                    title="乐高人工智能课程介绍",
                    url="https://www.yuque.com/example/lego-ai",
                    doc_id="101",
                )
            ],
            media=ChatMediaBundle(
                images=[
                    MediaItem(
                        url="/yuque/asset?t=abc",
                        title="课堂搭建图",
                        doc_title="乐高人工智能课程介绍",
                        doc_id="101",
                    )
                ],
                videos=[
                    MediaItem(
                        url="https://example.com/lego-demo.mp4",
                        title="课程演示视频",
                        doc_title="乐高人工智能课程介绍",
                        doc_id="101",
                    )
                ],
            ),
            debug={"mode": "mcp_get_doc", "doc_count": 1},
        )


class _FakeAdminVideoRepo:
    def __init__(self) -> None:
        self.calls: list[str | None] = []

    async def list_videos(self, *, scene_key: str | None = None):
        self.calls.append(scene_key)
        if scene_key != "school_ai_custom":
            return []

        class _Row:
            id = 88
            scene_key = "school_ai_custom"
            scene_name = "学校AI场景定制"
            title = "学校AI场景定制演示视频"
            original_filename = "school-demo.mp4"
            stored_filename = "20260615120000_school.mp4"
            file_path = "videos/school_ai_custom/20260615120000_school.mp4"
            file_url = "/admin-media/videos/school_ai_custom/20260615120000_school.mp4"
            mime_type = "video/mp4"
            file_size = 1024
            duration_seconds = None
            status = "active"
            created_at = ""
            updated_at = ""

        return [_Row()]


class _FakeAdminSceneIntroRepo:
    def __init__(
        self,
        *,
        intro_text: str = "这是后台维护的场景介绍。",
        decision_intro_text: str = "",
        user_intro_text: str = "",
    ) -> None:
        self.calls: list[str] = []
        self.intro_text = intro_text
        self.decision_intro_text = decision_intro_text
        self.user_intro_text = user_intro_text

    async def get_intro(self, *, scene_key: str):
        self.calls.append(scene_key)

        class _Row:
            def __init__(self, intro_text: str, decision_intro_text: str, user_intro_text: str) -> None:
                self.intro_text = intro_text
                self.decision_intro_text = decision_intro_text
                self.user_intro_text = user_intro_text

        return _Row(self.intro_text, self.decision_intro_text, self.user_intro_text)


@dataclass
class _FakeTocNode:
    uuid: str
    title: str
    level: int
    parent_uuid: str = ""
    type: str = "doc"
    url: str = ""
    doc_id: str = ""


class _FakeTocLoader:
    async def get_book_toc(self, *, book: str):  # noqa: ANN003
        return [
            _FakeTocNode(
                uuid=f"node-{idx}",
                title=f"目录节点{idx}",
                level=2,
                parent_uuid="root",
                doc_id=str(idx),
            )
            for idx in range(260)
        ]

    async def list_docs(self, **kwargs):  # noqa: ANN003
        raise AssertionError("TOC warmup should not fall back to list_docs when toc is available")


class _FakeSceneQueryRewriter:
    def __init__(self, rewritten_query: str = "智能招生 招生问答示例 招生流程 AI获客") -> None:
        self.rewritten_query = rewritten_query
        self.calls: list[dict[str, Any]] = []

    async def rewrite(self, *, question: str, scene: str, toc_nodes: list[dict[str, Any]]) -> str:
        self.calls.append(
            {
                "question": question,
                "scene": scene,
                "toc_titles": [str(item.get("title") or "") for item in toc_nodes],
            }
        )
        return self.rewritten_query


async def _collect_v5_events(
    orch: FriendDialogOrchestratorV5,
    *,
    question: str,
    session_id: str,
    scene: str,
    trigger_type: str,
    history: list[ChatMessageRow],
) -> list[dict[str, Any]]:
    return [
        item
        async for item in orch.answer_stream(
            question=question,
            session_id=session_id,
            scene=scene,
            trigger_type=trigger_type,
            history=history,
        )
    ]


_FAKE_TOC_NODES = [
    {"uuid": "root-ai", "title": "人工智能通识教育", "level": 1, "parent_uuid": "", "node_type": "title"},
    {"uuid": "ai-course", "title": "乐高人工智能课程介绍", "level": 2, "parent_uuid": "root-ai", "node_type": "doc"},
    {"uuid": "ai-class", "title": "课堂流程与作品展示", "level": 2, "parent_uuid": "root-ai", "node_type": "doc"},
    {"uuid": "ai-grade", "title": "适合年级与课时安排", "level": 2, "parent_uuid": "root-ai", "node_type": "doc"},
    {"uuid": "enroll", "title": "智能招生", "level": 1, "parent_uuid": "", "node_type": "title"},
    {"uuid": "enroll-faq", "title": "招生问答示例", "level": 2, "parent_uuid": "enroll", "node_type": "doc"},
]

_FAKE_CASE_TOC_NODES = [
    {"uuid": "platform", "title": "平台介绍", "level": 1, "parent_uuid": "", "node_type": "title"},
    {"uuid": "platform-ai", "title": "人工智能通识课程", "level": 2, "parent_uuid": "platform", "node_type": "doc"},
    {"uuid": "guide", "title": "使用指南", "level": 1, "parent_uuid": "", "node_type": "title"},
    {"uuid": "guide-ai", "title": "人工智能通识课程", "level": 2, "parent_uuid": "guide", "node_type": "doc", "url": YUQUE_SHARED_AI_COURSE_URL},
    {"uuid": "case-root", "title": "案例与社区", "level": 1, "parent_uuid": "", "node_type": "title"},
    {"uuid": "case-library", "title": "优秀案例库", "level": 2, "parent_uuid": "case-root", "node_type": "title"},
    {"uuid": "case-ai", "title": "人工智能通识课程", "level": 3, "parent_uuid": "case-library", "node_type": "doc", "url": YUQUE_SHARED_AI_CASE_URL},
    {"uuid": "case-pbl", "title": "跨学科项目式学习", "level": 3, "parent_uuid": "case-library", "node_type": "doc"},
    {"uuid": "case-representative", "title": "代表性案例", "level": 3, "parent_uuid": "case-library", "node_type": "doc"},
]


def test_public_yuque_share_url_uses_focus_node_url_dynamically() -> None:
    # 使用指南 / 优秀案例库 命中的聚焦节点：直接取其语雀 TOC 链接（目录更新自动跟随）
    assert (
        _public_yuque_share_url_for_focus(
            {"path": ["使用指南", "人工智能通识课程"], "url": YUQUE_SHARED_AI_COURSE_URL}
        )
        == YUQUE_SHARED_AI_COURSE_URL
    )
    assert (
        _public_yuque_share_url_for_focus(
            {"path": ["案例与社区", "优秀案例库", "人工智能通识课程"], "url": YUQUE_SHARED_AI_CASE_URL}
        )
        == YUQUE_SHARED_AI_CASE_URL
    )
    # 未带 singleDoc 的链接会自动补上单文档读取参数
    assert (
        _public_yuque_share_url_for_focus(
            {"path": ["使用指南", "学校AI场景定制"], "url": "https://www.yuque.com/suesun-yb1bi/sspenu/abcd1234"}
        )
        == "https://www.yuque.com/suesun-yb1bi/sspenu/abcd1234?singleDoc"
    )
    # 非「使用指南/优秀案例库」目录，或缺少链接时，不强制改写
    assert (
        _public_yuque_share_url_for_focus(
            {"path": ["平台介绍", "人工智能通识课程"], "url": YUQUE_SHARED_AI_COURSE_URL}
        )
        == ""
    )
    assert _public_yuque_share_url_for_focus({"path": ["使用指南", "人工智能通识课程"]}) == ""


@pytest.mark.asyncio
async def test_scene_trigger_uses_fixed_toc_mapping_and_reads_yuque_doc() -> None:
    yuque = _FakeYuqueSearch()
    deep_reader = _FakeDeepReader()
    rewriter = _FakeSceneQueryRewriter()
    orch = FriendDialogOrchestratorV5(
        generator=_FakeGenerator(),
        profile_repo=_FakeProfileRepo(),
        yuque_search=yuque,
        yuque_deep_reader=deep_reader,
        scene_query_rewriter=rewriter,
        profile_extractor=_FakeProfileExtractor(),
        toc_nodes=_FAKE_TOC_NODES,
    )

    events = [
        item
        async for item in orch.answer_stream(
            question="人工智能通识教育",
            session_id="sess_v5_scene",
            scene="人工智能通识教育",
            trigger_type="scene",
            history=[],
        )
    ]

    assert rewriter.calls == []
    assert deep_reader.calls == []
    assert deep_reader.node_calls[0]["question"] == "人工智能通识课程"
    assert deep_reader.node_calls[0]["node"]["title"] == "乐高人工智能课程介绍"
    assert yuque.calls == []
    assert not any("CatalogStateMachine" in str(item) for item in events)
    done = [item for item in events if item["event"] == "done"][0]["data"]
    # 首轮（T1）转化型标签受闸门拦截，不应出现「申请测试账号」
    assert trial_tag_for_scene("人工智能通识教育") not in done["tags"]
    assert done["answer"].startswith("我是小为")
    assert len(done["tags"]) == 3
    assert done["profile_fields"]["display_name"] == "赵老师"
    assert [source["source_type"] for source in done["sources"]] == ["web", "yuque"]
    assert done["sources"][1]["url"] == "https://www.yuque.com/example/lego-ai"
    assert done["debug"]["doc_deep_read_used"] is True
    assert done["debug"]["scene_query_rewrite"] == {
        "used": False,
        "rewritten_query": "人工智能通识课程",
        "skipped": "fixed_scene_toc_mapping",
    }
    assert done["debug"]["catalog_focus_node"]["title"] == "乐高人工智能课程介绍"


@pytest.mark.asyncio
async def test_tags_are_picked_from_yuque_toc_not_llm_random_tags() -> None:
    orch = FriendDialogOrchestratorV5(
        generator=_FakeGenerator(),
        profile_repo=_FakeProfileRepo(),
        profile_extractor=_FakeProfileExtractor(),
        toc_nodes=_FAKE_TOC_NODES,
    )

    events = [
        item
        async for item in orch.answer_stream(
            question="人工智能通识教育",
            session_id="sess_v5_toc_tags",
            scene="人工智能通识教育",
            trigger_type="scene",
            history=[],
        )
    ]

    done = [item for item in events if item["event"] == "done"][0]["data"]
    # 首轮：使用指南 + 当前聚焦文档的同级子目录探索（动态 TOC，统一问句风格），转化型标签受闸门拦截
    assert done["tags"] == [
        "想看看人工智能通识课程的产品的使用指南？",
        "想看看课堂流程与作品展示？",
        "想看看适合年级与课时安排？",
    ]
    assert "想看课程例子？" not in done["tags"]
    assert done["debug"]["catalog_tag_source"] == "fixed_v5_navigation"
    assert done["debug"]["catalog_focus_node"] == {
        "uuid": "ai-course",
        "title": "乐高人工智能课程介绍",
        "path": ["人工智能通识教育", "乐高人工智能课程介绍"],
    }


@pytest.mark.asyncio
async def test_first_turn_gates_conversion_tags_and_shows_subdir_exploration() -> None:
    orch = FriendDialogOrchestratorV5(
        generator=_FakeGenerator(),
        profile_repo=_FakeProfileRepo(),
        profile_extractor=_FakeProfileExtractor(),
        toc_nodes=_FAKE_TOC_NODES,
    )

    events = await _collect_v5_events(
        orch,
        question="人工智能通识教育",
        session_id="sess_v5_turn1_tags",
        scene="人工智能通识教育",
        trigger_type="scene",
        history=[],
    )

    done = [item for item in events if item["event"] == "done"][0]["data"]
    # T1：使用指南 + 子目录探索（统一问句）；案例库（≥T3）与测试账号（≥T4）被闸门拦截
    assert done["tags"] == [
        "想看看人工智能通识课程的产品的使用指南？",
        "想看看课堂流程与作品展示？",
        "想看看适合年级与课时安排？",
    ]
    assert done["debug"]["conversion_state"]["turn_index"] == 1
    assert done["debug"]["conversion_state"]["case_allowed"] is False
    assert done["debug"]["conversion_state"]["trial_allowed"] is False


@pytest.mark.asyncio
async def test_guide_tag_routes_to_usage_guide_and_next_tags_show_price() -> None:
    deep_reader = _FakeDeepReader()
    orch = FriendDialogOrchestratorV5(
        generator=_FakeGenerator(),
        profile_repo=_FakeProfileRepo(),
        yuque_deep_reader=deep_reader,
        profile_extractor=_FakeProfileExtractor(),
        toc_nodes=_FAKE_CASE_TOC_NODES,
    )

    events = await _collect_v5_events(
        orch,
        question="想看看人工智能通识课程的产品的使用指南？",
        session_id="sess_v5_turn2_tags",
        scene="人工智能通识教育",
        trigger_type="tag",
        history=[ChatMessageRow(role="user", content="人工智能通识教育", created_at="")],
    )

    done = [item for item in events if item["event"] == "done"][0]["data"]
    assert deep_reader.node_calls[0]["node"]["uuid"] == "guide-ai"
    assert done["tags"] == [
        "想要了解一下人工智能通识课程产品的价格？",
        "想看看人工智能通识课程的产品的优秀案例库？",
        "想申请测试账号，试一试人工智能通识课程的产品？",
    ]
    assert done["debug"]["tag_route"]["kind"] == "guide"
    assert done["debug"]["conversion_state"]["turn_index"] == 2


@pytest.mark.asyncio
async def test_price_tag_returns_handoff_without_yuque_read() -> None:
    deep_reader = _FakeDeepReader()
    orch = FriendDialogOrchestratorV5(
        generator=_FakeGenerator(),
        profile_repo=_FakeProfileRepo(),
        yuque_deep_reader=deep_reader,
        profile_extractor=_FakeProfileExtractor(),
        toc_nodes=_FAKE_TOC_NODES,
    )

    events = await _collect_v5_events(
        orch,
        question="想要了解一下人工智能通识课程产品的价格？",
        session_id="sess_v5_turn3_price",
        scene="人工智能通识教育",
        trigger_type="tag",
        history=[
            ChatMessageRow(role="user", content="人工智能通识教育", created_at=""),
            ChatMessageRow(role="user", content="想看看人工智能通识课程的产品的使用指南？", created_at=""),
        ],
    )

    done = [item for item in events if item["event"] == "done"][0]["data"]
    assert "方便留个微信或电话吗" in done["answer"]
    assert "方便留下您的称呼吗" not in done["answer"]
    assert deep_reader.calls == []
    assert deep_reader.node_calls == []
    assert done["debug"]["tag_route"]["kind"] == "price"
    assert done["debug"]["mcp_route"]["mode"] == "price_direct"


@pytest.mark.asyncio
async def test_followup_confirmation_uses_previous_followup_topic() -> None:
    deep_reader = _FakeDeepReader()
    orch = FriendDialogOrchestratorV5(
        generator=_FakeGenerator(),
        profile_repo=_FakeProfileRepo(),
        yuque_deep_reader=deep_reader,
        profile_extractor=_FakeProfileExtractor(),
        toc_nodes=_FAKE_TOC_NODES,
    )

    events = await _collect_v5_events(
        orch,
        question="需要",
        session_id="sess_v5_turn3_trial_price",
        scene="人工智能通识教育",
        trigger_type="manual",
        history=[
            ChatMessageRow(role="user", content="人工智能通识教育", created_at=""),
            ChatMessageRow(role="assistant", content="需要我和你详细介绍课堂流程与作品展示的内容吗？", created_at=""),
        ],
    )

    done = [item for item in events if item["event"] == "done"][0]["data"]
    assert deep_reader.node_calls[0]["question"] == "课堂流程与作品展示"
    assert done["debug"]["next_followup_topic"]
    assert done["debug"]["skill_route"]["skill_id"] == "smart-summary"


@pytest.mark.asyncio
async def test_scene_rewrite_maps_frontend_scene_alias_to_real_toc_node() -> None:
    toc_nodes = [
        {"uuid": "platform", "title": "平台介绍", "level": 1, "parent_uuid": "", "node_type": "title"},
        {
            "uuid": "ai-course",
            "title": "人工智能通识课程",
            "level": 2,
            "parent_uuid": "platform",
            "node_type": "doc",
        },
        {
            "uuid": "ai-lego",
            "title": "乐高人工智能课程",
            "level": 3,
            "parent_uuid": "ai-course",
            "node_type": "doc",
        },
        {
            "uuid": "ai-apple",
            "title": "苹果STEAM课程",
            "level": 3,
            "parent_uuid": "ai-course",
            "node_type": "doc",
        },
        {
            "uuid": "ai-sony",
            "title": "索尼人工智能课程",
            "level": 3,
            "parent_uuid": "ai-course",
            "node_type": "doc",
        },
    ]
    deep_reader = _FakeDeepReader()
    orch = FriendDialogOrchestratorV5(
        generator=_FakeGenerator(),
        profile_repo=_FakeProfileRepo(),
        yuque_deep_reader=deep_reader,
        scene_query_rewriter=_FakeSceneQueryRewriter("人工智能通识课程"),
        profile_extractor=_FakeProfileExtractor(),
        toc_nodes=toc_nodes,
    )

    events = [
        item
        async for item in orch.answer_stream(
            question="人工智能通识教育",
            session_id="sess_v5_scene_alias",
            scene="人工智能通识教育",
            trigger_type="scene",
            history=[],
        )
    ]

    assert deep_reader.node_calls[0]["node"]["title"] == "人工智能通识课程"
    done = [item for item in events if item["event"] == "done"][0]["data"]
    assert done["debug"]["catalog_focus_node"] == {
        "uuid": "ai-course",
        "title": "人工智能通识课程",
        "path": ["平台介绍", "人工智能通识课程"],
    }
    # T1：使用指南 + 聚焦文档下的子目录探索（动态 TOC 子节点，统一问句）
    assert done["tags"] == [
        "想看看人工智能通识课程的产品的使用指南？",
        "想看看乐高人工智能课程？",
        "想看看苹果STEAM课程？",
    ]


@pytest.mark.asyncio
async def test_scene_entry_recommends_own_subdir_not_sibling_scenes() -> None:
    # 平台介绍下有 4 个平级场景；进入「人工智能通识教育」时只推荐它自己目录下的子目录
    toc_nodes = [
        {"uuid": "platform", "title": "平台介绍", "level": 1, "parent_uuid": "", "node_type": "title"},
        {"uuid": "p-ai", "title": "人工智能通识课程", "level": 2, "parent_uuid": "platform", "node_type": "title"},
        {"uuid": "p-ai-tencent", "title": "腾讯青少年人工智能课程", "level": 3, "parent_uuid": "p-ai", "node_type": "doc"},
        {"uuid": "p-ai-ext", "title": "拓展课程", "level": 3, "parent_uuid": "p-ai", "node_type": "doc"},
        {"uuid": "p-pbl", "title": "跨学科项目式学习", "level": 2, "parent_uuid": "platform", "node_type": "doc"},
        {"uuid": "p-enroll", "title": "智能招生", "level": 2, "parent_uuid": "platform", "node_type": "doc"},
        {"uuid": "p-custom", "title": "学校AI场景定制", "level": 2, "parent_uuid": "platform", "node_type": "doc"},
    ]
    orch = FriendDialogOrchestratorV5(
        generator=_FakeGenerator(),
        profile_repo=_FakeProfileRepo(),
        profile_extractor=_FakeProfileExtractor(),
        toc_nodes=toc_nodes,
    )

    events = await _collect_v5_events(
        orch,
        question="人工智能通识教育",
        session_id="sess_v5_own_subdir",
        scene="人工智能通识教育",
        trigger_type="scene",
        history=[],
    )

    done = [item for item in events if item["event"] == "done"][0]["data"]
    assert done["tags"] == [
        "想看看人工智能通识课程的产品的使用指南？",
        "想看看腾讯青少年人工智能课程？",
        "想看看拓展课程？",
    ]
    # 关键：平级场景绝不出现在推荐标签中（含包装成问句的形式）
    for sibling in ("跨学科项目式学习", "智能招生", "学校AI场景定制"):
        assert not any(sibling in tag for tag in done["tags"])


@pytest.mark.asyncio
async def test_clicking_wrapped_subdir_tag_resolves_to_correct_toc_node() -> None:
    # 子目录标签统一为问句「想看看{标题}？」，点击后仍能定位回对应语雀目录节点并深读
    toc_nodes = [
        {"uuid": "platform", "title": "平台介绍", "level": 1, "parent_uuid": "", "node_type": "title"},
        {"uuid": "p-ai", "title": "人工智能通识课程", "level": 2, "parent_uuid": "platform", "node_type": "title"},
        {"uuid": "p-ai-ext", "title": "拓展课程", "level": 3, "parent_uuid": "p-ai", "node_type": "doc", "doc_id": "9001"},
    ]
    deep_reader = _FakeDeepReader()
    orch = FriendDialogOrchestratorV5(
        generator=_FakeGenerator(),
        profile_repo=_FakeProfileRepo(),
        yuque_deep_reader=deep_reader,
        profile_extractor=_FakeProfileExtractor(),
        toc_nodes=toc_nodes,
    )

    await _collect_v5_events(
        orch,
        question="想看看拓展课程？",
        session_id="sess_v5_wrapped_subdir_click",
        scene="人工智能通识教育",
        trigger_type="tag",
        history=[ChatMessageRow(role="user", content="人工智能通识教育", created_at="")],
    )

    assert deep_reader.node_calls[0]["node"]["title"] == "拓展课程"


@pytest.mark.asyncio
async def test_topical_container_tag_prefers_three_child_course_tags() -> None:
    toc_nodes = [
        {"uuid": "platform", "title": "平台介绍", "level": 1, "parent_uuid": "", "node_type": "title"},
        {"uuid": "p-ai", "title": "人工智能通识课程", "level": 2, "parent_uuid": "platform", "node_type": "title"},
        {"uuid": "p-ai-ext", "title": "拓展课程", "level": 3, "parent_uuid": "p-ai", "node_type": "doc"},
        {"uuid": "lego", "title": "乐高人工智能课程", "level": 4, "parent_uuid": "p-ai-ext", "node_type": "doc"},
        {"uuid": "apple", "title": "苹果STEAM课程", "level": 4, "parent_uuid": "p-ai-ext", "node_type": "doc"},
        {"uuid": "sony", "title": "索尼人工智能课程", "level": 4, "parent_uuid": "p-ai-ext", "node_type": "doc"},
    ]
    orch = FriendDialogOrchestratorV5(
        generator=_FakeGenerator(),
        profile_repo=_FakeProfileRepo(),
        profile_extractor=_FakeProfileExtractor(),
        toc_nodes=toc_nodes,
    )

    events = await _collect_v5_events(
        orch,
        question="想看看拓展课程？",
        session_id="sess_v5_expand_courses",
        scene="人工智能通识教育",
        trigger_type="tag",
        history=[ChatMessageRow(role="user", content="人工智能通识教育", created_at="")],
    )

    done = [item for item in events if item["event"] == "done"][0]["data"]
    assert done["debug"]["catalog_focus_node"]["title"] == "拓展课程"
    assert done["tags"] == [
        "想看看乐高人工智能课程？",
        "想看看苹果STEAM课程？",
        "想看看索尼人工智能课程？",
    ]


@pytest.mark.asyncio
async def test_topical_container_tag_does_not_auto_descend_to_first_course() -> None:
    class _RecordingDeepReader:
        def __init__(self) -> None:
            self.node_calls: list[dict[str, Any]] = []
            self.read_calls: list[str] = []

        async def read(self, *, question: str) -> FriendV5YuqueDeepReadResult:
            self.read_calls.append(question)
            return FriendV5YuqueDeepReadResult(debug={"mode": "search_fallback"})

        async def read_toc_node(self, *, node: dict[str, Any], question: str) -> FriendV5YuqueDeepReadResult:
            self.node_calls.append({"node": node, "question": question})
            return FriendV5YuqueDeepReadResult(debug={"mode": "toc_focus_read_miss"})

    toc_nodes = [
        {"uuid": "platform", "title": "平台介绍", "level": 1, "parent_uuid": "", "node_type": "title"},
        {"uuid": "p-ai", "title": "人工智能通识课程", "level": 2, "parent_uuid": "platform", "node_type": "title"},
        {"uuid": "p-ai-ext", "title": "拓展课程", "level": 3, "parent_uuid": "p-ai", "node_type": "doc"},
        {"uuid": "lego", "title": "乐高人工智能课程", "level": 4, "parent_uuid": "p-ai-ext", "node_type": "doc", "doc_id": "9001"},
        {"uuid": "apple", "title": "苹果STEAM课程", "level": 4, "parent_uuid": "p-ai-ext", "node_type": "doc", "doc_id": "9002"},
        {"uuid": "sony", "title": "索尼人工智能课程", "level": 4, "parent_uuid": "p-ai-ext", "node_type": "doc", "doc_id": "9003"},
    ]
    generator = _FakeGenerator()
    deep_reader = _RecordingDeepReader()
    orch = FriendDialogOrchestratorV5(
        generator=generator,
        profile_repo=_FakeProfileRepo(),
        yuque_deep_reader=deep_reader,
        profile_extractor=_FakeProfileExtractor(),
        toc_nodes=toc_nodes,
    )

    await _collect_v5_events(
        orch,
        question="想看看拓展课程？",
        session_id="sess_v5_expand_courses_prompt",
        scene="人工智能通识教育",
        trigger_type="tag",
        history=[ChatMessageRow(role="user", content="人工智能通识教育", created_at="")],
    )

    assert deep_reader.node_calls == []
    assert deep_reader.read_calls == []
    assert "当前命中的是语雀目录「拓展课程」" in generator.user_prompt
    assert "这不是单个课程文档，不要直接默认展开成其中某一个课程" in generator.user_prompt
    assert "1. 乐高人工智能课程" in generator.user_prompt
    assert "2. 苹果STEAM课程" in generator.user_prompt
    assert "3. 索尼人工智能课程" in generator.user_prompt


@pytest.mark.asyncio
async def test_clicking_wrapped_long_subdir_tag_descends_to_readable_doc() -> None:
    class _DescendAwareDeepReader:
        def __init__(self) -> None:
            self.node_calls: list[dict[str, Any]] = []

        async def read(self, *, question: str) -> FriendV5YuqueDeepReadResult:
            return FriendV5YuqueDeepReadResult(debug={"mode": "search_fallback"})

        async def read_toc_node(self, *, node: dict[str, Any], question: str) -> FriendV5YuqueDeepReadResult:
            self.node_calls.append({"node": node, "question": question})
            if not str(node.get("doc_id") or "").strip():
                return FriendV5YuqueDeepReadResult(debug={"mode": "toc_focus_read_miss"})
            return FriendV5YuqueDeepReadResult(
                used=True,
                prompt_block="【语雀文档深读】\n标题：腾讯青少年人工智能课程\n正文摘录：课程目标、课时安排、作品示例。",
                sources=[
                    FriendV5SourceItem(
                        source_type="yuque",
                        title="腾讯青少年人工智能课程",
                        url="https://www.yuque.com/example/tencent-ai-course",
                        doc_id=str(node.get("doc_id") or ""),
                    )
                ],
                debug={"mode": "mcp_get_doc", "doc_count": 1},
            )

    toc_nodes = [
        {"uuid": "platform", "title": "平台介绍", "level": 1, "parent_uuid": "", "node_type": "title"},
        {"uuid": "p-ai", "title": "人工智能通识课程", "level": 2, "parent_uuid": "platform", "node_type": "title"},
        {"uuid": "p-ai-tencent", "title": "腾讯青少年人工智能课程", "level": 3, "parent_uuid": "p-ai", "node_type": "title"},
        {"uuid": "p-ai-tencent-doc", "title": "腾讯青少年人工智能课程详情", "level": 4, "parent_uuid": "p-ai-tencent", "node_type": "doc", "doc_id": "9002"},
    ]
    deep_reader = _DescendAwareDeepReader()
    orch = FriendDialogOrchestratorV5(
        generator=_FakeGenerator(),
        profile_repo=_FakeProfileRepo(),
        yuque_deep_reader=deep_reader,
        profile_extractor=_FakeProfileExtractor(),
        toc_nodes=toc_nodes,
    )

    await _collect_v5_events(
        orch,
        question="想看看腾讯青少年人工智能课程？",
        session_id="sess_v5_wrapped_long_subdir_click",
        scene="人工智能通识教育",
        trigger_type="tag",
        history=[ChatMessageRow(role="user", content="人工智能通识教育", created_at="")],
    )

    assert deep_reader.node_calls
    assert deep_reader.node_calls[0]["node"]["title"] == "腾讯青少年人工智能课程详情"
    assert str(deep_reader.node_calls[0]["node"]["doc_id"]) == "9002"


@pytest.mark.asyncio
async def test_leaf_focus_falls_back_to_three_strong_sibling_tags_without_far_jump() -> None:
    toc_nodes = [
        {"uuid": "root-ai", "title": "人工智能通识教育", "level": 1, "parent_uuid": "", "node_type": "title"},
        {"uuid": "ai-lego", "title": "乐高人工智能课程", "level": 2, "parent_uuid": "root-ai", "node_type": "doc"},
        {"uuid": "ai-apple", "title": "苹果STEAM课程", "level": 2, "parent_uuid": "root-ai", "node_type": "doc"},
        {"uuid": "ai-sony", "title": "索尼人工智能课程", "level": 2, "parent_uuid": "root-ai", "node_type": "doc"},
        {"uuid": "ai-tencent", "title": "腾讯青少年人工智能课程", "level": 2, "parent_uuid": "root-ai", "node_type": "doc"},
        {"uuid": "guide", "title": "使用指南", "level": 1, "parent_uuid": "", "node_type": "title"},
        {"uuid": "guide-enroll", "title": "智能招生操作说明", "level": 2, "parent_uuid": "guide", "node_type": "doc"},
    ]
    orch = FriendDialogOrchestratorV5(
        generator=_FakeGenerator(),
        profile_repo=_FakeProfileRepo(),
        profile_extractor=_FakeProfileExtractor(),
        toc_nodes=toc_nodes,
    )

    events = [
        item
        async for item in orch.answer_stream(
            question="苹果STEAM课程",
            session_id="sess_v5_leaf_tags",
            scene="人工智能通识教育",
            trigger_type="tag",
            history=[],
        )
    ]

    done = [item for item in events if item["event"] == "done"][0]["data"]
    assert done["debug"]["catalog_focus_node"] == {
        "uuid": "ai-apple",
        "title": "苹果STEAM课程",
        "path": ["人工智能通识教育", "苹果STEAM课程"],
    }
    # T1：使用指南 + 同级强相关产品（子目录探索，统一问句），不跨到「使用指南」下的远节点
    assert done["tags"] == [
        "想看看人工智能通识课程的产品的使用指南？",
        "想看看乐高人工智能课程？",
        "想看看索尼人工智能课程？",
    ]
    assert not any("智能招生操作说明" in tag for tag in done["tags"])


@pytest.mark.asyncio
async def test_tag_trigger_deep_reads_matched_toc_node_when_reader_available() -> None:
    deep_reader = _FakeDeepReader()
    orch = FriendDialogOrchestratorV5(
        generator=_FakeGenerator(),
        profile_repo=_FakeProfileRepo(),
        yuque_search=_FakeYuqueSearch(),
        yuque_deep_reader=deep_reader,
        profile_extractor=_FakeProfileExtractor(),
        toc_nodes=_FAKE_TOC_NODES,
    )

    events = [
        item
        async for item in orch.answer_stream(
            question="乐高人工智能课程介绍",
            session_id="sess_v5_tag_deep_read",
            scene="人工智能通识教育",
            trigger_type="tag",
            history=[],
        )
    ]

    assert deep_reader.calls == []
    assert deep_reader.node_calls[0]["node"]["title"] == "乐高人工智能课程介绍"
    assert deep_reader.node_calls[0]["question"] == "乐高人工智能课程介绍"
    done = [item for item in events if item["event"] == "done"][0]["data"]
    assert done["debug"]["doc_deep_read_used"] is True
    assert done["debug"]["catalog_focus_node"]["title"] == "乐高人工智能课程介绍"


@pytest.mark.asyncio
async def test_case_tag_uses_product_from_tag_text_not_sidebar_scene() -> None:
    deep_reader = _FakeDeepReader()
    orch = FriendDialogOrchestratorV5(
        generator=_FakeGenerator(),
        profile_repo=_FakeProfileRepo(),
        yuque_deep_reader=deep_reader,
        profile_extractor=_FakeProfileExtractor(),
        toc_nodes=_FAKE_CASE_TOC_NODES,
    )

    await _collect_v5_events(
        orch,
        question=case_tag_for_scene("跨学科项目化学习"),
        session_id="sess_v5_case_cross_product",
        scene="人工智能通识教育",
        trigger_type="tag",
        history=[],
    )

    assert deep_reader.node_calls[0]["node"]["uuid"] == "case-pbl"


@pytest.mark.asyncio
async def test_case_tag_without_toc_node_returns_fixed_fallback() -> None:
    generator = _FakeGenerator()
    orch = FriendDialogOrchestratorV5(
        generator=generator,
        profile_repo=_FakeProfileRepo(),
        yuque_deep_reader=_FakeDeepReader(),
        profile_extractor=_FakeProfileExtractor(),
        toc_nodes=_FAKE_CASE_TOC_NODES,
    )

    events = await _collect_v5_events(
        orch,
        question=case_tag_for_scene("学校AI场景定制"),
        session_id="sess_v5_case_miss",
        scene="人工智能通识教育",
        trigger_type="tag",
        history=[],
    )

    done = [item for item in events if item["event"] == "done"][0]["data"]
    assert done["answer"] == _CASE_KB_FALLBACK_ANSWER
    assert done["fallback_used"] is True
    assert done["debug"]["case_toc_miss"] is True
    assert done["debug"]["doc_deep_read_used"] is False
    assert done["debug"]["case_kb_fallback"] is True
    assert done["debug"]["web_search_fallback_enabled"] is False
    assert not hasattr(generator, "enable_search")


@pytest.mark.asyncio
async def test_case_tag_with_empty_yuque_body_returns_fixed_fallback() -> None:
    generator = _FakeGenerator()
    deep_reader = _FakeDeepReader(used=False)
    orch = FriendDialogOrchestratorV5(
        generator=generator,
        profile_repo=_FakeProfileRepo(),
        yuque_deep_reader=deep_reader,
        profile_extractor=_FakeProfileExtractor(),
        toc_nodes=_FAKE_CASE_TOC_NODES,
    )

    events = await _collect_v5_events(
        orch,
        question=case_tag_for_scene("人工智能通识教育"),
        session_id="sess_v5_case_empty_body",
        scene="人工智能通识教育",
        trigger_type="tag",
        history=[],
    )

    done = [item for item in events if item["event"] == "done"][0]["data"]
    assert deep_reader.node_calls
    assert done["answer"] == _CASE_KB_FALLBACK_ANSWER
    assert done["debug"]["case_kb_fallback"] is True
    assert done["debug"]["doc_deep_read"]["empty_body"] is True
    assert done["debug"]["web_search_fallback_enabled"] is False


@pytest.mark.asyncio
async def test_product_case_tag_switches_to_case_library_not_platform_or_guide() -> None:
    deep_reader = _FakeDeepReader()
    orch = FriendDialogOrchestratorV5(
        generator=_FakeGenerator(),
        profile_repo=_FakeProfileRepo(),
        yuque_search=_FakeYuqueSearch(),
        yuque_deep_reader=deep_reader,
        profile_extractor=_FakeProfileExtractor(),
        toc_nodes=_FAKE_CASE_TOC_NODES,
    )

    events = [
        item
        async for item in orch.answer_stream(
            question="产品案例",
            session_id="sess_v5_product_case",
            scene="人工智能通识教育",
            trigger_type="tag",
            history=[
                ChatMessageRow(role="user", content="乐高人工智能课程", created_at=""),
                ChatMessageRow(role="assistant", content="这里适合通识课程场景。", created_at=""),
            ],
        )
    ]

    assert deep_reader.calls == []
    assert deep_reader.node_calls[0]["node"]["uuid"] == "case-ai"
    done = [item for item in events if item["event"] == "done"][0]["data"]
    assert done["debug"]["case_branch_used"] is True
    assert done["debug"]["catalog_focus_node"]["path"] == ["案例与社区", "优秀案例库", "人工智能通识课程"]
    assert done["debug"]["doc_deep_read_used"] is True
    yuque_sources = [source for source in done["sources"] if source["source_type"] == "yuque"]
    assert yuque_sources
    assert {source["url"] for source in yuque_sources} == {YUQUE_SHARED_AI_CASE_URL}
    assert done["media"]["images"][0]["url"] == "/yuque/asset?t=abc"
    assert "平台介绍" not in done["debug"]["catalog_focus_node"]["path"]
    assert "使用指南" not in done["debug"]["catalog_focus_node"]["path"]


@pytest.mark.asyncio
async def test_product_case_tag_prefers_same_name_case_before_generic_direction() -> None:
    toc_nodes = [
        *_FAKE_CASE_TOC_NODES,
        {"uuid": "case-lego", "title": "乐高人工智能课程", "level": 3, "parent_uuid": "case-library", "node_type": "doc"},
    ]
    deep_reader = _FakeDeepReader()
    orch = FriendDialogOrchestratorV5(
        generator=_FakeGenerator(),
        profile_repo=_FakeProfileRepo(),
        yuque_deep_reader=deep_reader,
        profile_extractor=_FakeProfileExtractor(),
        toc_nodes=toc_nodes,
    )

    await _collect_v5_events(
        orch,
        question="产品案例",
        session_id="sess_v5_same_name_case",
        scene="人工智能通识教育",
        trigger_type="tag",
        history=[ChatMessageRow(role="user", content="乐高人工智能课程", created_at="")],
    )

    assert deep_reader.node_calls[0]["node"]["uuid"] == "case-lego"


@pytest.mark.asyncio
async def test_product_case_tag_falls_back_to_representative_case() -> None:
    toc_nodes = [
        node
        for node in _FAKE_CASE_TOC_NODES
        if node["uuid"] not in {"case-ai", "case-pbl"}
    ]
    deep_reader = _FakeDeepReader()
    orch = FriendDialogOrchestratorV5(
        generator=_FakeGenerator(),
        profile_repo=_FakeProfileRepo(),
        yuque_deep_reader=deep_reader,
        profile_extractor=_FakeProfileExtractor(),
        toc_nodes=toc_nodes,
    )

    await _collect_v5_events(
        orch,
        question="产品案例",
        session_id="sess_v5_representative_case",
        scene="人工智能通识教育",
        trigger_type="tag",
        history=[ChatMessageRow(role="user", content="乐高人工智能课程", created_at="")],
    )

    assert deep_reader.node_calls[0]["node"]["uuid"] == "case-representative"


@pytest.mark.asyncio
async def test_qa_service_warmup_keeps_full_toc_tree() -> None:
    original_scope = settings.yuque_scope
    object.__setattr__(settings, "yuque_scope", "fake/repo")
    service = QAService.__new__(QAService)
    service._yuque_loader = _FakeTocLoader()
    service._guide_doc_titles = []
    service._guide_toc_nodes = []
    service._guide_titles_refreshed_at = 0.0

    try:
        await service._warmup_guide_doc_titles()
    finally:
        object.__setattr__(settings, "yuque_scope", original_scope)

    assert len(service._guide_toc_nodes) == 260
    assert len(service._guide_doc_titles) == 80
    assert service._guide_toc_nodes[-1]["title"] == "目录节点259"


@pytest.mark.asyncio
async def test_tag_trigger_queries_yuque_url_without_document_body() -> None:
    yuque = _FakeYuqueSearch()
    generator = _FakeGenerator()
    orch = FriendDialogOrchestratorV5(
        generator=generator,
        profile_repo=_FakeProfileRepo(),
        yuque_search=yuque,
        profile_extractor=_FakeProfileExtractor(),
    )

    events = [
        item
        async for item in orch.answer_stream(
            question="乐高人工智能课程",
            session_id="sess_v5_tag",
            scene="人工智能通识教育",
            trigger_type="tag",
            history=[],
        )
    ]

    assert yuque.calls == ["乐高人工智能课程"]
    assert "【语雀文档深读】" not in generator.user_prompt
    done = [item for item in events if item["event"] == "done"][0]["data"]
    assert [source["source_type"] for source in done["sources"]] == ["web", "yuque"]


@pytest.mark.asyncio
async def test_scene_first_turn_returns_uploaded_admin_video() -> None:
    admin_videos = _FakeAdminVideoRepo()
    orch = FriendDialogOrchestratorV5(
        generator=_FakeGenerator(),
        profile_repo=_FakeProfileRepo(),
        yuque_search=_FakeYuqueSearch(),
        profile_extractor=_FakeProfileExtractor(),
        admin_video_repository=admin_videos,
    )

    events = [
        item
        async for item in orch.answer_stream(
            question="学校AI场景定制",
            session_id="sess_v5_admin_video",
            scene="学校AI场景定制",
            trigger_type="scene",
            history=[],
        )
    ]

    done = [item for item in events if item["event"] == "done"][0]["data"]
    event_names = [item["event"] for item in events]
    assert event_names.index("media_preview") < event_names.index("token")
    preview = [item for item in events if item["event"] == "media_preview"][0]["data"]
    assert preview["media"]["videos"][0]["url"] == "/admin-media/videos/school_ai_custom/20260615120000_school.mp4"
    assert preview["media_display_mode"] == "before_answer"
    assert admin_videos.calls == ["school_ai_custom"]
    assert done["media"]["videos"][0]["url"] == "/admin-media/videos/school_ai_custom/20260615120000_school.mp4"
    assert done["media"]["videos"][0]["title"] == "学校AI场景定制演示视频"
    assert done["debug"]["admin_scene_video_count"] == 1
    assert done["debug"]["media_display_mode"] == "before_answer"
    assert "可以先看这段学校AI场景定制的演示视频" in done["debug"]["media_intro"]


@pytest.mark.asyncio
async def test_scene_trigger_returns_uploaded_admin_video_even_with_history() -> None:
    admin_videos = _FakeAdminVideoRepo()
    orch = FriendDialogOrchestratorV5(
        generator=_FakeGenerator(),
        profile_repo=_FakeProfileRepo(),
        yuque_search=_FakeYuqueSearch(),
        profile_extractor=_FakeProfileExtractor(),
        admin_video_repository=admin_videos,
    )

    events = [
        item
        async for item in orch.answer_stream(
            question="学校AI场景定制",
            session_id="sess_v5_admin_video_with_history",
            scene="学校AI场景定制",
            trigger_type="scene",
            history=[
                ChatMessageRow(role="user", content="智能招生", created_at=""),
                ChatMessageRow(role="assistant", content="智能招生介绍。", created_at=""),
            ],
        )
    ]

    done = [item for item in events if item["event"] == "done"][0]["data"]
    assert admin_videos.calls == ["school_ai_custom"]
    assert done["media"]["videos"][0]["url"] == "/admin-media/videos/school_ai_custom/20260615120000_school.mp4"
    assert done["debug"]["admin_scene_video_count"] == 1


@pytest.mark.asyncio
async def test_admin_scene_intro_is_injected_into_system_prompt() -> None:
    generator = _FakeGenerator()
    scene_intro_repo = _FakeAdminSceneIntroRepo(
        intro_text="IDEAS-PBL 强调项目生成、过程数据留存和智能评价闭环。",
        decision_intro_text="决策者更关注学校级落地与数据分析。",
        user_intro_text="使用者更关注备课效率和课堂组织。",
    )
    profile_repo = _FakeProfileRepo()
    profile_repo.profile = _FakeProfile(visitor_type="institution_decision_maker")
    orch = FriendDialogOrchestratorV5(
        generator=generator,
        profile_repo=profile_repo,
        yuque_search=_FakeYuqueSearch(),
        profile_extractor=_FakeProfileExtractor(),
        admin_scene_intro_repository=scene_intro_repo,
    )

    await _collect_v5_events(
        orch,
        question="跨学科项目化学习",
        session_id="sess_v5_admin_scene_intro",
        scene="跨学科项目化学习",
        trigger_type="scene",
        history=[],
    )

    assert scene_intro_repo.calls == ["project_based_learning"]
    assert "【后台维护的当前场景产品介绍】" in generator.system_prompt
    assert "IDEAS-PBL 强调项目生成、过程数据留存和智能评价闭环。" in generator.system_prompt
    assert "当前识别到的访客身份：决策者（校长/负责人）" in generator.system_prompt
    assert "【本轮优先采用口径】\n决策者更关注学校级落地与数据分析。" in generator.system_prompt


@pytest.mark.asyncio
async def test_manual_specific_doc_question_uses_deep_reader_and_returns_media() -> None:
    deep_reader = _FakeDeepReader()
    generator = _FakeGenerator()
    orch = FriendDialogOrchestratorV5(
        generator=generator,
        profile_repo=_FakeProfileRepo(),
        yuque_search=_FakeYuqueSearch(),
        yuque_deep_reader=deep_reader,
        profile_extractor=_FakeProfileExtractor(),
    )

    events = [
        item
        async for item in orch.answer_stream(
            question="帮我总结乐高人工智能课程介绍那篇语雀文档",
            session_id="sess_v5_deep",
            scene="人工智能通识教育",
            trigger_type="manual",
            history=[],
        )
    ]

    assert deep_reader.calls == ["帮我总结乐高人工智能课程介绍那篇语雀文档"]
    assert "【语雀文档深读】" in generator.user_prompt
    done = [item for item in events if item["event"] == "done"][0]["data"]
    assert done["debug"]["doc_deep_read_used"] is True
    assert done["media"]["images"][0]["url"] == "/yuque/asset?t=abc"
    assert done["media"]["videos"][0]["url"] == "https://example.com/lego-demo.mp4"
    assert [source["source_type"] for source in done["sources"]] == ["web", "yuque"]


@pytest.mark.asyncio
async def test_manual_toc_matched_question_deep_reads_toc_node_and_returns_media() -> None:
    deep_reader = _FakeDeepReader()
    generator = _FakeGenerator()
    orch = FriendDialogOrchestratorV5(
        generator=generator,
        profile_repo=_FakeProfileRepo(),
        yuque_search=_FakeYuqueSearch(),
        yuque_deep_reader=deep_reader,
        profile_extractor=_FakeProfileExtractor(),
        toc_nodes=_FAKE_TOC_NODES,
    )

    events = [
        item
        async for item in orch.answer_stream(
            question="乐高人工智能课程里，老师培训和教师支持怎么做？",
            session_id="sess_v5_manual_toc",
            scene="人工智能通识教育",
            trigger_type="manual",
            history=[],
        )
    ]

    assert deep_reader.calls == []
    assert deep_reader.node_calls[0]["node"]["title"] == "乐高人工智能课程介绍"
    done = [item for item in events if item["event"] == "done"][0]["data"]
    assert done["debug"]["doc_deep_read_used"] is True
    assert done["media"]["images"][0]["url"] == "/yuque/asset?t=abc"
    assert done["media"]["videos"][0]["url"] == "https://example.com/lego-demo.mp4"


class _FakeGeneratorWithInlineLink:
    async def stream(self, *, system_prompt: str, user_prompt: str, enable_search: bool = False):
        self.user_prompt = user_prompt
        yield FriendV5StreamEvent.token(
            "这份指南帮老师快速上手。\n\nwww.yuque.com/suesun-yb1bi/sspenu/sbdx665n47rz9rt5\n"
        )
        yield FriendV5StreamEvent.token("[TAGS]想看课程例子？\n想了解适合年级？\n想看看落地方式？[END_TAGS]")


@pytest.mark.asyncio
async def test_tag_click_cross_scene_product_redirects_instead_of_case_library() -> None:
    deep_reader = _FakeDeepReader()
    generator = _FakeGenerator()
    orch = FriendDialogOrchestratorV5(
        generator=generator,
        profile_repo=_FakeProfileRepo(),
        yuque_deep_reader=deep_reader,
        profile_extractor=_FakeProfileExtractor(),
        toc_nodes=_FAKE_CASE_TOC_NODES,
    )

    events = await _collect_v5_events(
        orch,
        question="跨学科项目式学习",
        session_id="sess_v5_tag_case_cont",
        scene="人工智能通识教育",
        trigger_type="tag",
        history=[
            ChatMessageRow(role="user", content="人工智能通识教育", created_at=""),
            ChatMessageRow(role="user", content="想看看人工智能通识课程的产品的优秀案例库？", created_at=""),
        ],
    )

    assert deep_reader.node_calls == []
    assert not getattr(generator, "user_prompt", None)
    done = [item for item in events if item["event"] == "done"][0]["data"]
    assert done["debug"]["cross_scene_redirect"] is True
    assert "请先在左侧点击对应场景" in done["answer"]
    assert "跨学科项目式学习" in done["answer"]


@pytest.mark.asyncio
async def test_manual_cross_scene_product_redirects_instead_of_case_library() -> None:
    deep_reader = _FakeDeepReader()
    orch = FriendDialogOrchestratorV5(
        generator=_FakeGenerator(),
        profile_repo=_FakeProfileRepo(),
        yuque_deep_reader=deep_reader,
        profile_extractor=_FakeProfileExtractor(),
        toc_nodes=_FAKE_CASE_TOC_NODES,
    )

    events = await _collect_v5_events(
        orch,
        question="跨学科项目式学习",
        session_id="sess_v5_manual_case_cont",
        scene="人工智能通识教育",
        trigger_type="manual",
        history=[
            ChatMessageRow(role="user", content="人工智能通识教育", created_at=""),
            ChatMessageRow(role="user", content="想看看人工智能通识课程的产品的优秀案例库？", created_at=""),
        ],
    )

    assert deep_reader.node_calls == []
    done = [item for item in events if item["event"] == "done"][0]["data"]
    assert done["debug"]["cross_scene_redirect"] is True
    assert "请先在左侧点击对应场景" in done["answer"]


@pytest.mark.asyncio
async def test_same_scene_product_after_case_history_routes_to_case_library() -> None:
    deep_reader = _FakeDeepReader()
    generator = _FakeGenerator()
    orch = FriendDialogOrchestratorV5(
        generator=generator,
        profile_repo=_FakeProfileRepo(),
        yuque_deep_reader=deep_reader,
        profile_extractor=_FakeProfileExtractor(),
        toc_nodes=_FAKE_CASE_TOC_NODES,
    )

    events = await _collect_v5_events(
        orch,
        question="人工智能通识课程",
        session_id="sess_v5_same_scene_case_cont",
        scene="人工智能通识教育",
        trigger_type="manual",
        history=[
            ChatMessageRow(role="user", content="人工智能通识教育", created_at=""),
            ChatMessageRow(role="user", content="想看看人工智能通识课程的产品的优秀案例库？", created_at=""),
        ],
    )

    assert deep_reader.node_calls[0]["node"]["uuid"] == "case-ai"
    assert "优秀案例库模式" in generator.user_prompt
    done = [item for item in events if item["event"] == "done"][0]["data"]
    assert done["debug"]["case_product_switch"] is True
    assert done["debug"]["catalog_focus_node"]["path"] == ["案例与社区", "优秀案例库", "人工智能通识课程"]


@pytest.mark.asyncio
async def test_scene_trigger_after_case_history_routes_to_platform_intro() -> None:
    toc_nodes = [
        *_FAKE_CASE_TOC_NODES,
        {
            "uuid": "platform-pbl",
            "title": "跨学科项目式学习",
            "level": 2,
            "parent_uuid": "platform",
            "node_type": "doc",
        },
    ]
    deep_reader = _FakeDeepReader()
    orch = FriendDialogOrchestratorV5(
        generator=_FakeGenerator(),
        profile_repo=_FakeProfileRepo(),
        yuque_deep_reader=deep_reader,
        profile_extractor=_FakeProfileExtractor(),
        toc_nodes=toc_nodes,
    )

    events = await _collect_v5_events(
        orch,
        question="跨学科项目化学习",
        session_id="sess_v5_scene_case_cont",
        scene="跨学科项目化学习",
        trigger_type="scene",
        history=[
            ChatMessageRow(role="user", content="人工智能通识教育", created_at=""),
            ChatMessageRow(role="user", content="想看看人工智能通识课程的产品的优秀案例库？", created_at=""),
        ],
    )

    assert deep_reader.node_calls[0]["node"]["uuid"] == "platform-pbl"
    done = [item for item in events if item["event"] == "done"][0]["data"]
    assert done["debug"]["scene_case_continuation"] is False
    assert done["debug"]["catalog_focus_node"]["path"] == ["平台介绍", "跨学科项目式学习"]
    assert done["debug"]["mcp_route"]["mode"] == "scene_toc"


@pytest.mark.asyncio
async def test_scene_trigger_after_guide_history_routes_to_platform_intro() -> None:
    toc_nodes = [
        *_FAKE_CASE_TOC_NODES,
        {
            "uuid": "platform-pbl",
            "title": "跨学科项目式学习",
            "level": 2,
            "parent_uuid": "platform",
            "node_type": "doc",
        },
    ]
    deep_reader = _FakeDeepReader()
    orch = FriendDialogOrchestratorV5(
        generator=_FakeGenerator(),
        profile_repo=_FakeProfileRepo(),
        yuque_deep_reader=deep_reader,
        profile_extractor=_FakeProfileExtractor(),
        toc_nodes=toc_nodes,
    )

    events = await _collect_v5_events(
        orch,
        question="跨学科项目化学习",
        session_id="sess_v5_scene_guide_cont",
        scene="跨学科项目化学习",
        trigger_type="scene",
        history=[
            ChatMessageRow(role="user", content="人工智能通识教育", created_at=""),
            ChatMessageRow(
                role="user",
                content=guide_tag_for_scene("人工智能通识教育"),
                created_at="",
            ),
        ],
    )

    assert deep_reader.node_calls[0]["node"]["uuid"] == "platform-pbl"
    done = [item for item in events if item["event"] == "done"][0]["data"]
    assert done["debug"]["scene_guide_continuation"] is False
    assert done["debug"]["catalog_focus_node"]["path"] == ["平台介绍", "跨学科项目式学习"]
    assert "使用指南" not in done["debug"]["catalog_focus_node"]["path"]
    assert done["debug"]["mcp_route"]["mode"] == "scene_toc"


@pytest.mark.asyncio
async def test_case_history_recommends_other_platform_intro_products() -> None:
    toc_nodes = [
        *_FAKE_CASE_TOC_NODES,
        {
            "uuid": "platform-pbl",
            "title": "跨学科项目式学习",
            "level": 2,
            "parent_uuid": "platform",
            "node_type": "doc",
        },
        {
            "uuid": "platform-custom",
            "title": "学校AI场景定制",
            "level": 2,
            "parent_uuid": "platform",
            "node_type": "doc",
        },
    ]
    orch = FriendDialogOrchestratorV5(
        generator=_FakeGenerator(),
        profile_repo=_FakeProfileRepo(),
        yuque_deep_reader=_FakeDeepReader(),
        profile_extractor=_FakeProfileExtractor(),
        toc_nodes=toc_nodes,
        admin_video_repository=_FakeAdminVideoRepo(),
    )

    events = await _collect_v5_events(
        orch,
        question=case_tag_for_scene("人工智能通识教育"),
        session_id="sess_v5_case_explore_tags",
        scene="人工智能通识教育",
        trigger_type="tag",
        history=[ChatMessageRow(role="user", content="人工智能通识教育", created_at="")],
    )

    done = [item for item in events if item["event"] == "done"][0]["data"]
    assert done["tags"][0] == explore_product_tag_for_title("跨学科项目式学习")
    assert done["debug"]["conversion_state"]["stage"] == "case_to_explore_products"


@pytest.mark.asyncio
async def test_after_explore_product_tag_restores_normal_rhythm() -> None:
    toc_nodes = [
        *_FAKE_CASE_TOC_NODES,
        {
            "uuid": "platform-pbl",
            "title": "跨学科项目式学习",
            "level": 2,
            "parent_uuid": "platform",
            "node_type": "doc",
        },
        {
            "uuid": "platform-custom",
            "title": "学校AI场景定制",
            "level": 2,
            "parent_uuid": "platform",
            "node_type": "doc",
        },
    ]
    orch = FriendDialogOrchestratorV5(
        generator=_FakeGenerator(),
        profile_repo=_FakeProfileRepo(),
        yuque_deep_reader=_FakeDeepReader(),
        profile_extractor=_FakeProfileExtractor(),
        toc_nodes=toc_nodes,
        admin_video_repository=_FakeAdminVideoRepo(),
    )

    events = await _collect_v5_events(
        orch,
        question=explore_product_tag_for_title("学校AI场景定制"),
        session_id="sess_v5_after_explore_product",
        scene="人工智能通识教育",
        trigger_type="tag",
        history=[
            ChatMessageRow(role="user", content=case_tag_for_scene("人工智能通识教育"), created_at=""),
            ChatMessageRow(role="assistant", content="这里是一个落地案例。", created_at=""),
            ChatMessageRow(
                role="user",
                content=explore_product_tag_for_title("学校AI场景定制"),
                created_at="",
            ),
        ],
    )

    done = [item for item in events if item["event"] == "done"][0]["data"]
    # 探索产品后恢复常规漏斗节奏（指南/案例为主），而非案例库分支的横向探索列表
    assert done["tags"][0] == guide_tag_for_scene("学校AI场景定制")
    assert done["tags"][1] == case_tag_for_scene("学校AI场景定制")
    # 学校AI场景定制在该 TOC 下无自有子目录，兜底用转化型标签补齐（真实库有子目录时走闸门节奏）
    assert done["tags"][2] == trial_tag_for_scene("学校AI场景定制")
    # 关键：绝不把平级场景（如「跨学科项目式学习」）当作子目录推荐
    assert "跨学科项目式学习" not in done["tags"]
    assert explore_product_tag_for_title("跨学科项目式学习") not in done["tags"]
    assert done["debug"]["conversion_state"]["stage"] == "conversion_unlocked"
    assert done["debug"]["conversion_state"]["turn_index"] == 3
    assert done["debug"]["doc_deep_read_used"] is True
    assert done["debug"]["media_suppressed"] is True
    assert done["media"]["images"] == []
    assert done["media"]["videos"][0]["url"] == "/admin-media/videos/school_ai_custom/20260615120000_school.mp4"
    assert done["debug"]["admin_scene_video_count"] == 1
    assert done["debug"]["media_display_mode"] == "before_answer"
    assert "可以先看这段学校AI场景定制的演示视频" in done["debug"]["media_intro"]


@pytest.mark.asyncio
async def test_scene_trigger_without_case_history_keeps_platform_intro_route() -> None:
    deep_reader = _FakeDeepReader()
    orch = FriendDialogOrchestratorV5(
        generator=_FakeGenerator(),
        profile_repo=_FakeProfileRepo(),
        yuque_deep_reader=deep_reader,
        profile_extractor=_FakeProfileExtractor(),
        toc_nodes=_FAKE_CASE_TOC_NODES,
    )

    events = await _collect_v5_events(
        orch,
        question="人工智能通识教育",
        session_id="sess_v5_scene_no_case",
        scene="人工智能通识教育",
        trigger_type="scene",
        history=[],
    )

    done = [item for item in events if item["event"] == "done"][0]["data"]
    assert done["debug"]["scene_case_continuation"] is False
    assert done["debug"]["mcp_route"]["mode"] == "scene_toc"
    assert done["debug"]["doc_deep_read_used"] is True
    assert done["debug"]["media_suppressed"] is True
    assert done["media"]["images"] == []
    assert done["media"]["videos"] == []


@pytest.mark.asyncio
async def test_web_search_fallback_enabled_only_when_deep_read_missing() -> None:
    original = settings.chat_v5_web_search_enabled
    object.__setattr__(settings, "chat_v5_web_search_enabled", True)
    try:
        gen_with_deep_read = _FakeGenerator()
        orch_with = FriendDialogOrchestratorV5(
            generator=gen_with_deep_read,
            profile_repo=_FakeProfileRepo(),
            yuque_deep_reader=_FakeDeepReader(),
            profile_extractor=_FakeProfileExtractor(),
            toc_nodes=_FAKE_TOC_NODES,
        )
        await _collect_v5_events(
            orch_with,
            question="人工智能通识教育",
            session_id="sess_v5_search_off",
            scene="人工智能通识教育",
            trigger_type="scene",
            history=[],
        )
        assert gen_with_deep_read.enable_search is False

        gen_without_deep_read = _FakeGenerator()
        orch_without = FriendDialogOrchestratorV5(
            generator=gen_without_deep_read,
            profile_repo=_FakeProfileRepo(),
            profile_extractor=_FakeProfileExtractor(),
        )
        events = await _collect_v5_events(
            orch_without,
            question="智能招生怎么做？",
            session_id="sess_v5_search_on",
            scene="智能招生",
            trigger_type="manual",
            history=[],
        )
        assert gen_without_deep_read.enable_search is True
        done = [item for item in events if item["event"] == "done"][0]["data"]
        assert done["debug"]["web_search_fallback_enabled"] is True
    finally:
        object.__setattr__(settings, "chat_v5_web_search_enabled", original)


@pytest.mark.asyncio
async def test_guide_answer_strips_inline_links_and_appends_friendly_hint() -> None:
    deep_reader = _FakeDeepReader()
    orch = FriendDialogOrchestratorV5(
        generator=_FakeGeneratorWithInlineLink(),
        profile_repo=_FakeProfileRepo(),
        yuque_deep_reader=deep_reader,
        profile_extractor=_FakeProfileExtractor(),
        toc_nodes=_FAKE_CASE_TOC_NODES,
    )

    events = await _collect_v5_events(
        orch,
        question="想看看人工智能通识课程的产品的使用指南？",
        session_id="sess_v5_guide_hint",
        scene="人工智能通识教育",
        trigger_type="tag",
        history=[ChatMessageRow(role="user", content="人工智能通识教育", created_at="")],
    )

    done = [item for item in events if item["event"] == "done"][0]["data"]
    assert "www.yuque.com/example/lego-ai" not in done["answer"]
    assert f"[人工智能通识课程使用指南]({YUQUE_SHARED_AI_COURSE_URL})" in done["answer"]
    yuque_sources = [source for source in done["sources"] if source["source_type"] == "yuque"]
    assert yuque_sources
    assert {source["url"] for source in yuque_sources} == {YUQUE_SHARED_AI_COURSE_URL}


@pytest.mark.asyncio
async def test_followup_question_skips_discussed_child_and_mentions_sibling() -> None:
    toc_nodes = [
        {"uuid": "platform", "title": "平台介绍", "level": 1, "parent_uuid": "", "node_type": "title"},
        {"uuid": "ai-course", "title": "人工智能通识课程", "level": 2, "parent_uuid": "platform", "node_type": "doc"},
        {"uuid": "ai-lego", "title": "乐高人工智能课程", "level": 3, "parent_uuid": "ai-course", "node_type": "doc"},
        {"uuid": "ai-apple", "title": "苹果STEAM课程", "level": 3, "parent_uuid": "ai-course", "node_type": "doc"},
        {"uuid": "ai-sony", "title": "索尼人工智能课程", "level": 3, "parent_uuid": "ai-course", "node_type": "doc"},
    ]
    deep_reader = _FakeDeepReader()
    orch = FriendDialogOrchestratorV5(
        generator=_FakeGenerator(),
        profile_repo=_FakeProfileRepo(),
        yuque_deep_reader=deep_reader,
        profile_extractor=_FakeProfileExtractor(),
        toc_nodes=toc_nodes,
    )

    # 已介绍过乐高：追问应换成下一个未介绍的子话题
    events = await _collect_v5_events(
        orch,
        question="人工智能通识教育",
        session_id="sess_v5_followup_skip",
        scene="人工智能通识教育",
        trigger_type="scene",
        history=[
            ChatMessageRow(role="user", content="人工智能通识教育", created_at=""),
            ChatMessageRow(role="assistant", content="需要我和你详细介绍乐高人工智能课程的内容吗？", created_at=""),
        ],
    )
    done = [item for item in events if item["event"] == "done"][0]["data"]
    assert done["debug"]["next_followup_topic"] == "苹果STEAM课程"

    # 聚焦叶子节点时：仍会算出同级推荐方向，但不再硬拼固定尾巴
    events = await _collect_v5_events(
        orch,
        question="乐高人工智能课程",
        session_id="sess_v5_followup_sibling",
        scene="人工智能通识教育",
        trigger_type="tag",
        history=[],
    )
    done = [item for item in events if item["event"] == "done"][0]["data"]
    assert done["debug"]["followup_sibling_topic"] == "苹果STEAM课程"
    assert "如果您对苹果STEAM课程也感兴趣，也可以为您介绍。" not in done["answer"]
    assert "跨学科项目式学习也感兴趣" not in done["answer"]


@pytest.mark.asyncio
async def test_repeated_identity_question_is_stripped_when_recent_history_already_asked() -> None:
    class _IdentityRepeatingGenerator:
        async def stream(self, *, system_prompt: str, user_prompt: str, enable_search: bool = False):
            yield FriendV5StreamEvent.token("先简单说一下。\n您这边是校长/负责人，还是老师、家长、学生呢？我按您的关注点来介绍。")
            yield FriendV5StreamEvent.token("[SOURCES]\n[/SOURCES]\n[TAGS][END_TAGS]")

    orch = FriendDialogOrchestratorV5(
        generator=_IdentityRepeatingGenerator(),
        profile_repo=_FakeProfileRepo(),
        yuque_deep_reader=_FakeDeepReader(),
        profile_extractor=_FakeProfileExtractor(),
        toc_nodes=_FAKE_TOC_NODES,
    )

    events = await _collect_v5_events(
        orch,
        question="想看看人工智能通识课程的产品的使用指南？",
        session_id="sess_v5_identity_repeat",
        scene="人工智能通识教育",
        trigger_type="tag",
        history=[
            ChatMessageRow(role="assistant", content="您这边是校长/负责人，还是老师、家长、学生呢？我按您的关注点来介绍。", created_at=""),
        ],
    )

    done = [item for item in events if item["event"] == "done"][0]["data"]
    assert done["answer"].startswith("先简单说一下。")
    assert "您这边是校长/负责人，还是老师、家长、学生呢？我按您的关注点来介绍。" not in done["answer"]


@pytest.mark.asyncio
async def test_inline_repeated_identity_question_is_also_stripped() -> None:
    class _InlineIdentityRepeatingGenerator:
        async def stream(self, *, system_prompt: str, user_prompt: str, enable_search: bool = False):
            yield FriendV5StreamEvent.token(
                "目前提供的资料里暂时没有具体的操作使用指南细节。您这边是校长/负责人，还是老师、家长、学生呢？我按您的关注点来介绍。"
            )
            yield FriendV5StreamEvent.token("[SOURCES]\n[/SOURCES]\n[TAGS][END_TAGS]")

    orch = FriendDialogOrchestratorV5(
        generator=_InlineIdentityRepeatingGenerator(),
        profile_repo=_FakeProfileRepo(),
        yuque_deep_reader=_FakeDeepReader(),
        profile_extractor=_FakeProfileExtractor(),
        toc_nodes=_FAKE_TOC_NODES,
    )

    events = await _collect_v5_events(
        orch,
        question="想看看人工智能通识课程的产品的使用指南？",
        session_id="sess_v5_identity_repeat_inline",
        scene="人工智能通识教育",
        trigger_type="tag",
        history=[
            ChatMessageRow(role="assistant", content="您这边是校长/负责人，还是老师、家长、学生呢？我按您的关注点来介绍。", created_at=""),
        ],
    )

    done = [item for item in events if item["event"] == "done"][0]["data"]
    assert "您这边是校长/负责人，还是老师、家长、学生呢？我按您的关注点来介绍。" not in done["answer"]
    assert "具体操作我把指南放下面，您可以先看。" in done["answer"]


@pytest.mark.asyncio
async def test_tag_followup_strips_identity_reask_variant_after_identity_already_asked() -> None:
    class _IdentityVariantRepeatingGenerator:
        async def stream(self, *, system_prompt: str, user_prompt: str, enable_search: bool = False):
            yield FriendV5StreamEvent.token(
                "目前提供的资料中暂无具体的操作使用指南细节。不过我可以先为您梳理一下智能招生的核心功能亮点，或者您方便告知一下您的身份（如校长或老师）吗？这样我能更针对性地介绍它如何帮您减轻招生咨询的负担。"
            )
            yield FriendV5StreamEvent.token("[SOURCES]\n[/SOURCES]\n[TAGS][END_TAGS]")

    orch = FriendDialogOrchestratorV5(
        generator=_IdentityVariantRepeatingGenerator(),
        profile_repo=_FakeProfileRepo(),
        yuque_deep_reader=_FakeDeepReader(),
        profile_extractor=_FakeProfileExtractor(),
        toc_nodes=_FAKE_TOC_NODES,
    )

    events = await _collect_v5_events(
        orch,
        question="想看看人工智能通识课程的产品的使用指南？",
        session_id="sess_v5_identity_repeat_variant",
        scene="人工智能通识教育",
        trigger_type="tag",
        history=[
            ChatMessageRow(role="assistant", content="您这边是校长/负责人，还是老师、家长、学生呢？我按您的关注点来介绍。", created_at=""),
        ],
    )

    done = [item for item in events if item["event"] == "done"][0]["data"]
    assert "您方便告知一下您的身份" not in done["answer"]
    assert "具体操作我把指南放下面，您可以先看。" in done["answer"]


@pytest.mark.asyncio
async def test_guide_link_only_answer_avoids_detail_missing_phrase_and_repeated_identity_reask() -> None:
    class _GuideLinkOnlyGenerator:
        async def stream(self, *, system_prompt: str, user_prompt: str, enable_search: bool = False):
            yield FriendV5StreamEvent.token(
                "您好，我是小为。您好，我是小为。目前资料里没包含具体的操作指南细节，这块通常是根据学校实际招生流程来配置的。"
                "您这边是校长/负责人，还是负责招生的老师呢？我按您的角色看看怎么介绍更合适。\n"
                "[SOURCES]\n"
                "https://www.yuque.com/suesun-yb1bi/sspenu/pmg3pix4w4e6g1zd?singleDoc#%20《智能招生》\n"
                "[/SOURCES]\n"
                "[TAGS][END_TAGS]"
            )

    orch = FriendDialogOrchestratorV5(
        generator=_GuideLinkOnlyGenerator(),
        profile_repo=_FakeProfileRepo(),
        profile_extractor=_FakeProfileExtractor(),
        toc_nodes=_FAKE_TOC_NODES,
    )

    events = await _collect_v5_events(
        orch,
        question="想看看使用指南？",
        session_id="sess_v5_guide_link_only",
        scene="智能招生",
        trigger_type="tag",
        history=[
            ChatMessageRow(role="assistant", content="您这边是校长/负责人，还是老师、家长、学生呢？我按您的关注点来介绍。", created_at=""),
        ],
    )

    done = [item for item in events if item["event"] == "done"][0]["data"]
    assert "目前资料里没包含具体的操作指南细节" not in done["answer"]
    assert "负责招生的老师呢" not in done["answer"]
    assert done["answer"].count("您好，我是小为") <= 1
    assert "具体操作我把指南放下面，您可以先看" in done["answer"]


@pytest.mark.asyncio
async def test_internal_timeout_status_line_is_stripped_from_answer() -> None:
    class _TimeoutLeakGenerator:
        async def stream(self, *, system_prompt: str, user_prompt: str, enable_search: bool = False):
            yield FriendV5StreamEvent.token(
                "刚才文档读取超时，没法直接调出详细案例页。\n"
                "不过这块落地很广，像上海宝山世外也有相关实践。[SOURCES]\n[/SOURCES]\n[TAGS][END_TAGS]"
            )

    orch = FriendDialogOrchestratorV5(
        generator=_TimeoutLeakGenerator(),
        profile_repo=_FakeProfileRepo(),
        profile_extractor=_FakeProfileExtractor(),
        toc_nodes=_FAKE_TOC_NODES,
    )

    events = await _collect_v5_events(
        orch,
        question="想看人工智能通识教育案例",
        session_id="sess_v5_timeout_leak",
        scene="人工智能通识教育",
        trigger_type="manual",
        history=[],
    )

    done = [item for item in events if item["event"] == "done"][0]["data"]
    assert "刚才文档读取超时" not in done["answer"]
    assert "详细案例页" not in done["answer"]
    assert "这块落地很广，像上海宝山世外也有相关实践。" in done["answer"]


@pytest.mark.asyncio
async def test_generic_fallback_line_and_assumptive_school_wording_are_softened() -> None:
    class _FallbackLeakGenerator:
        async def stream(self, *, system_prompt: str, user_prompt: str, enable_search: bool = False):
            yield FriendV5StreamEvent.token(
                "这块我先按目前能确认的信息和您讲。\n"
                "咱们学校大概有多少老师专门负责这块接待工作？[SOURCES]\n[/SOURCES]\n[TAGS][END_TAGS]"
            )

    orch = FriendDialogOrchestratorV5(
        generator=_FallbackLeakGenerator(),
        profile_repo=_FakeProfileRepo(),
        profile_extractor=_FakeProfileExtractor(),
        toc_nodes=_FAKE_TOC_NODES,
    )

    events = await _collect_v5_events(
        orch,
        question="招生接待压力很大",
        session_id="sess_v5_fallback_soften",
        scene="智能招生",
        trigger_type="manual",
        history=[],
    )

    done = [item for item in events if item["event"] == "done"][0]["data"]
    assert "这块我先按目前能确认的信息和您讲" not in done["answer"]
    assert "咱们学校" not in done["answer"]
    assert "您这边更想先看试点怎么跑，还是先看老师要配合到什么程度" in done["answer"]


@pytest.mark.asyncio
async def test_smart_enrollment_teacher_count_question_is_softened_into_choice_style_followup() -> None:
    class _TeacherCountQuestionGenerator:
        async def stream(self, *, system_prompt: str, user_prompt: str, enable_search: bool = False):
            yield FriendV5StreamEvent.token(
                "明白，这块确实最耗精力。智能招生能自动承接大部分基础咨询，像作息、费用这些常见问题它都能秒回，把您从重复劳动里解放出来。"
                "您平时晚上大概要处理多少条这类消息？[SOURCES]\n[/SOURCES]\n[TAGS][END_TAGS]"
            )

    orch = FriendDialogOrchestratorV5(
        generator=_TeacherCountQuestionGenerator(),
        profile_repo=_FakeProfileRepo(),
        profile_extractor=_FakeProfileExtractor(),
        toc_nodes=_FAKE_TOC_NODES,
    )

    events = await _collect_v5_events(
        orch,
        question="我是招生老师，晚上家长咨询很多",
        session_id="sess_v5_teacher_count_soften",
        scene="智能招生",
        trigger_type="manual",
        history=[],
    )

    done = [item for item in events if item["event"] == "done"][0]["data"]
    assert "多少条这类消息" not in done["answer"]
    assert "更想先看它能替老师省下哪些重复回复，还是先看怎么试用" in done["answer"]


@pytest.mark.asyncio
async def test_smart_enrollment_decision_maker_staff_count_question_is_softened_into_trial_path_followup() -> None:
    class _DecisionMakerCountQuestionGenerator:
        async def stream(self, *, system_prompt: str, user_prompt: str, enable_search: bool = False):
            yield FriendV5StreamEvent.token(
                "这类系统通常适合咨询量大、重复问题多的学校，能显著减轻招生办压力。落地一般不需复杂开发，主要是把学校常见的问答资料整理好导入即可，前期准备比较轻。"
                "您学校目前大概有多少老师专门负责接待家长咨询呢？[SOURCES]\n[/SOURCES]\n[TAGS][END_TAGS]"
            )

    orch = FriendDialogOrchestratorV5(
        generator=_DecisionMakerCountQuestionGenerator(),
        profile_repo=_FakeProfileRepo(),
        profile_extractor=_FakeProfileExtractor(),
        toc_nodes=_FAKE_TOC_NODES,
    )

    events = await _collect_v5_events(
        orch,
        question="我是校长，想了解智能招生前期落地麻烦吗",
        session_id="sess_v5_decision_count_soften",
        scene="智能招生",
        trigger_type="manual",
        history=[],
    )

    done = [item for item in events if item["event"] == "done"][0]["data"]
    assert "多少老师专门负责接待家长咨询" not in done["answer"]
    assert "更想先看试点怎么跑，还是先看老师要配合到什么程度" in done["answer"]


@pytest.mark.asyncio
async def test_web_sources_event_wins_over_placeholder_sources_block() -> None:
    orch = FriendDialogOrchestratorV5(
        generator=_FakeGeneratorWithWebSources(),
        profile_repo=_FakeProfileRepo(),
        profile_extractor=_FakeProfileExtractor(),
    )

    events = [
        item
        async for item in orch.answer_stream(
            question="智能招生",
            session_id="sess_v5_web_sources",
            scene="智能招生",
            trigger_type="manual",
            history=[],
        )
    ]

    done = [item for item in events if item["event"] == "done"][0]["data"]
    assert done["answer"] == "我是小为，先说结论。"
    assert done["sources"] == [
        {
            "source_type": "web",
            "title": "真实联网来源",
            "url": "https://www.youweiai.com/news",
            "snippet": "联网搜索摘要",
            "index": 1,
            "doc_id": None,
        }
    ]
