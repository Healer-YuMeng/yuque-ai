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
from app.service.friend_dialog_orchestrator_v5 import FriendDialogOrchestratorV5
from app.service.friend_v5_yuque_deep_reader import FriendV5YuqueDeepReadResult


@dataclass
class _FakeProfile:
    display_name: str = ""
    org_name: str = ""
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
    async def stream(self, *, system_prompt: str, user_prompt: str):
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        yield FriendV5StreamEvent.token("我是小为，先帮你把重点拎出来。")
        yield FriendV5StreamEvent.token(
            "[SOURCES]\n"
            "https://www.youweiai.com/web\n"
            "[/SOURCES]\n"
            "[TAGS]想看课程例子？\n想了解适合年级？\n想看看落地方式？[END_TAGS]"
        )


class _FakeGeneratorWithWebSources:
    async def stream(self, *, system_prompt: str, user_prompt: str):
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
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
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.node_calls: list[dict[str, Any]] = []

    async def read(self, *, question: str) -> FriendV5YuqueDeepReadResult:
        self.calls.append(question)
        return self._result()

    async def read_toc_node(self, *, node: dict[str, Any], question: str) -> FriendV5YuqueDeepReadResult:
        self.node_calls.append({"node": node, "question": question})
        return self._result()

    def _result(self) -> FriendV5YuqueDeepReadResult:
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
    {"uuid": "guide-ai", "title": "人工智能通识课程", "level": 2, "parent_uuid": "guide", "node_type": "doc"},
    {"uuid": "case-root", "title": "案例与社区", "level": 1, "parent_uuid": "", "node_type": "title"},
    {"uuid": "case-library", "title": "优秀案例库", "level": 2, "parent_uuid": "case-root", "node_type": "title"},
    {"uuid": "case-ai", "title": "人工智能通识课程", "level": 3, "parent_uuid": "case-library", "node_type": "doc"},
    {"uuid": "case-pbl", "title": "跨学科项目式学习", "level": 3, "parent_uuid": "case-library", "node_type": "doc"},
    {"uuid": "case-representative", "title": "代表性案例", "level": 3, "parent_uuid": "case-library", "node_type": "doc"},
]


@pytest.mark.asyncio
async def test_scene_trigger_rewrites_query_and_reads_yuque_doc() -> None:
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

    assert [item["scene"] for item in rewriter.calls] == ["人工智能通识教育"]
    assert deep_reader.calls == []
    assert deep_reader.node_calls[0]["question"] == "智能招生 招生问答示例 招生流程 AI获客"
    assert deep_reader.node_calls[0]["node"]["title"] == "人工智能通识教育"
    assert yuque.calls == []
    assert not any("CatalogStateMachine" in str(item) or "trial" in str(item) for item in events)
    done = [item for item in events if item["event"] == "done"][0]["data"]
    assert done["answer"].startswith("我是小为")
    assert len(done["tags"]) == 3
    assert done["profile_fields"]["display_name"] == "赵老师"
    assert [source["source_type"] for source in done["sources"]] == ["web", "yuque"]
    assert done["sources"][1]["url"] == "https://www.yuque.com/example/lego-ai"
    assert done["debug"]["doc_deep_read_used"] is True
    assert done["debug"]["scene_query_rewrite"]["rewritten_query"] == "智能招生 招生问答示例 招生流程 AI获客"
    assert done["debug"]["catalog_focus_node"]["title"] == "人工智能通识教育"


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
    assert done["tags"] == ["智能招生", "乐高人工智能课程介绍", "课堂流程与作品展示"]
    assert "想看课程例子？" not in done["tags"]
    assert done["debug"]["catalog_tag_source"] == "yuque_toc"
    assert done["debug"]["catalog_focus_node"] == {
        "uuid": "root-ai",
        "title": "人工智能通识教育",
        "path": ["人工智能通识教育"],
    }


@pytest.mark.asyncio
async def test_first_turn_keeps_only_directory_content_tags() -> None:
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
    assert "产品价格" not in done["tags"]
    assert "如何申请内测" not in done["tags"]
    assert "产品案例" not in done["tags"]
    assert done["debug"]["conversion_state"]["turn_index"] == 1


@pytest.mark.asyncio
async def test_second_turn_uses_two_content_tags_plus_product_case() -> None:
    toc_nodes = [
        {"uuid": "root-ai", "title": "人工智能通识教育", "level": 1, "parent_uuid": "", "node_type": "title"},
        {"uuid": "ai-lego", "title": "乐高人工智能课程", "level": 2, "parent_uuid": "root-ai", "node_type": "doc"},
        {"uuid": "ai-apple", "title": "苹果STEAM课程", "level": 2, "parent_uuid": "root-ai", "node_type": "doc"},
        {"uuid": "ai-sony", "title": "索尼人工智能课程", "level": 2, "parent_uuid": "root-ai", "node_type": "doc"},
        {"uuid": "ai-tencent", "title": "腾讯青少年人工智能课程", "level": 2, "parent_uuid": "root-ai", "node_type": "doc"},
    ]
    orch = FriendDialogOrchestratorV5(
        generator=_FakeGenerator(),
        profile_repo=_FakeProfileRepo(),
        profile_extractor=_FakeProfileExtractor(),
        toc_nodes=toc_nodes,
    )

    events = await _collect_v5_events(
        orch,
        question="乐高人工智能课程",
        session_id="sess_v5_turn2_tags",
        scene="人工智能通识教育",
        trigger_type="tag",
        history=[ChatMessageRow(role="user", content="人工智能通识教育", created_at="")],
    )

    done = [item for item in events if item["event"] == "done"][0]["data"]
    assert done["tags"] == ["苹果STEAM课程", "索尼人工智能课程", "产品案例"]
    assert "产品价格" not in done["tags"]
    assert "如何申请内测" not in done["tags"]
    assert done["debug"]["conversion_state"]["turn_index"] == 2


@pytest.mark.asyncio
async def test_third_turn_price_intent_adds_price_without_removing_all_content_tags() -> None:
    orch = FriendDialogOrchestratorV5(
        generator=_FakeGenerator(),
        profile_repo=_FakeProfileRepo(),
        profile_extractor=_FakeProfileExtractor(),
        toc_nodes=_FAKE_TOC_NODES,
    )

    events = await _collect_v5_events(
        orch,
        question="乐高人工智能课程大概多少钱？",
        session_id="sess_v5_turn3_price",
        scene="人工智能通识教育",
        trigger_type="manual",
        history=[
            ChatMessageRow(role="user", content="人工智能通识教育", created_at=""),
            ChatMessageRow(role="assistant", content="可以先看课程方向。", created_at=""),
            ChatMessageRow(role="user", content="乐高人工智能课程", created_at=""),
        ],
    )

    done = [item for item in events if item["event"] == "done"][0]["data"]
    assert "产品价格" in done["tags"]
    assert "如何申请内测" not in done["tags"]
    assert any(tag not in {"产品价格", "如何申请内测", "产品案例"} for tag in done["tags"])
    assert done["debug"]["conversion_state"]["conversion_intents"] == ["price"]


@pytest.mark.asyncio
async def test_third_turn_trial_and_price_intents_keep_one_content_tag() -> None:
    orch = FriendDialogOrchestratorV5(
        generator=_FakeGenerator(),
        profile_repo=_FakeProfileRepo(),
        profile_extractor=_FakeProfileExtractor(),
        toc_nodes=_FAKE_TOC_NODES,
    )

    events = await _collect_v5_events(
        orch,
        question="我想了解价格，也想申请内测账号",
        session_id="sess_v5_turn3_trial_price",
        scene="人工智能通识教育",
        trigger_type="manual",
        history=[
            ChatMessageRow(role="user", content="人工智能通识教育", created_at=""),
            ChatMessageRow(role="assistant", content="可以先看课程方向。", created_at=""),
            ChatMessageRow(role="user", content="乐高人工智能课程", created_at=""),
        ],
    )

    done = [item for item in events if item["event"] == "done"][0]["data"]
    conversion_tags = [tag for tag in done["tags"] if tag in {"产品价格", "如何申请内测"}]
    content_tags = [tag for tag in done["tags"] if tag not in {"产品价格", "如何申请内测", "产品案例"}]
    assert conversion_tags == ["产品价格", "如何申请内测"]
    assert len(content_tags) == 1
    assert done["debug"]["conversion_state"]["conversion_intents"] == ["price", "trial"]


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
    assert done["tags"] == ["乐高人工智能课程", "苹果STEAM课程", "索尼人工智能课程"]


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
    assert done["tags"] == ["乐高人工智能课程", "索尼人工智能课程", "腾讯青少年人工智能课程"]
    assert "智能招生操作说明" not in done["tags"]


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
    assert "语雀正文" not in generator.user_prompt
    done = [item for item in events if item["event"] == "done"][0]["data"]
    assert [source["source_type"] for source in done["sources"]] == ["web", "yuque"]


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
