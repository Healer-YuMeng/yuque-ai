from __future__ import annotations

import pytest

from app.conversation.chat_display import display_name_for_chat
from app.conversation.profile_extractor import ProfileExtractor
from app.conversation.toc_catalog import CatalogNode
from app.conversation.v4_lead_outreach import _build_v4_nudge_text, _lead_complete
from app.db.repositories import ChatMessageRow
from app.db.profile_repository import ChatSessionProfile
from app.service.media_answer_orchestrator import _DocContext, collect_media_from_doc_contexts
from app.service.qa_service import _visitor_profile_parts
from app.service.sales_dialog_orchestrator_v4 import (
    _build_repeat_feedback_apology,
    _build_v4_prompt,
    _detect_repeat_feedback_field,
    _media_scope_docs,
    _should_skip_mcp_search,
    _should_show_media,
)


def test_display_name_for_chat_appends_teacher_suffix() -> None:
    profile = ChatSessionProfile(
        session_id="s1",
        display_name="赵",
        visitor_type="teacher",
        org_name="",
        interests={},
        focused_doc_ids=[],
    )
    assert display_name_for_chat(profile) == "赵老师"


def test_display_name_for_chat_uses_full_self_introduction_without_truncation() -> None:
    profile = ChatSessionProfile(
        session_id="s1",
        display_name="育才中学的赵老师",
        visitor_type="teacher",
        org_name="育才中学",
        interests={},
        focused_doc_ids=[],
    )
    assert display_name_for_chat(profile) == "赵老师"


def test_visitor_profile_parts_filters_invalid_lead_name() -> None:
    profile = ChatSessionProfile(
        session_id="s1",
        display_name="",
        visitor_type="teacher",
        org_name="",
        interests={"_lead": {"name": "给小学", "org_name": "", "contact_value": ""}},
        focused_doc_ids=[],
    )
    parts = _visitor_profile_parts(profile)
    assert parts["name"] == ""


def test_v4_nudge_asks_one_lead_field() -> None:
    text, asked = _build_v4_nudge_text(profile=None, lead_meta={}, session_meta={})
    assert "称呼" in text
    assert "单位" not in text
    assert "联系方式" not in text
    assert "测试账号" not in text
    assert asked == "name"


def test_v4_nudge_uses_full_display_name() -> None:
    profile = ChatSessionProfile(
        session_id="s1",
        display_name="赵",
        visitor_type="teacher",
        org_name="",
        interests={},
        focused_doc_ids=[],
    )
    text, _ = _build_v4_nudge_text(profile=profile, lead_meta={}, session_meta={})
    assert text.startswith("赵老师，")


def test_v4_nudge_skips_already_asked_org_name() -> None:
    profile = ChatSessionProfile(
        session_id="s1",
        display_name="赵老师",
        visitor_type="teacher",
        org_name="",
        interests={},
        focused_doc_ids=[],
    )
    text, asked = _build_v4_nudge_text(
        profile=profile,
        lead_meta={},
        session_meta={"asked_fields": ["org_name"]},
    )
    assert "单位或学校" not in text
    assert asked == "contact"


def test_v4_nudge_contact_offer_provides_value_before_contact() -> None:
    profile = ChatSessionProfile(
        session_id="s1",
        display_name="赵老师",
        visitor_type="teacher",
        org_name="育才中学",
        interests={},
        focused_doc_ids=[],
    )
    text, asked = _build_v4_nudge_text(profile=profile, lead_meta={}, session_meta={})
    assert asked == "contact"
    assert "完整案例资料" in text
    assert "申请测试账号" in text
    assert "发给您" in text
    assert "手机号" not in text
    assert "微信" not in text


def test_v4_nudge_does_not_actively_collect_interested_product() -> None:
    profile = ChatSessionProfile(
        session_id="s1",
        display_name="赵老师",
        visitor_type="teacher",
        org_name="育才中学",
        interests={},
        focused_doc_ids=[],
    )
    text, asked = _build_v4_nudge_text(
        profile=profile,
        lead_meta={"contact_value": "18012345678"},
        session_meta={},
    )
    assert text == ""
    assert asked == ""


def test_v4_lead_complete_requires_only_three_collected_fields() -> None:
    profile = ChatSessionProfile(
        session_id="s1",
        display_name="赵老师",
        visitor_type="teacher",
        org_name="育才中学",
        interests={},
        focused_doc_ids=[],
    )
    assert _lead_complete({"contact_value": "18012345678"}, profile) is True


@pytest.mark.asyncio
async def test_profile_extractor_zhao_teacher_full_name() -> None:
    ex = ProfileExtractor()
    upd = await ex.extract_update(question="你好，我是赵老师", history=[], current_profile=None)
    assert upd.display_name == "赵老师"


def test_should_not_show_media_for_trial_only() -> None:
    node = CatalogNode(
        uuid="u1",
        title="乐高 AI 课程",
        level=2,
        parent_uuid="p",
        node_type="DOC",
        url=None,
        doc_id=1,
        path_titles=["课程", "乐高 AI 课程"],
    )
    primary = _DocContext(
        doc_id="1",
        title="乐高 AI 课程",
        url="",
        snippet="",
        body="![图](https://cdn.nlark.com/yuque/0/test.png)",
    )
    assert _should_show_media(question="我想要测试账号", node=node, primary=primary) is False


def test_collect_media_scoped_to_primary_doc_title() -> None:
    a = _DocContext(doc_id="1", title="文档A", url="", snippet="", body="![a](https://cdn.nlark.com/yuque/0/a.png)")
    b = _DocContext(doc_id="2", title="文档B", url="", snippet="", body="![b](https://cdn.nlark.com/yuque/0/b.png)")
    bundle = collect_media_from_doc_contexts(
        [a, b],
        question="平台介绍",
        max_images=3,
        max_videos=0,
        primary_doc_title="文档A",
    )
    assert len(bundle.images) == 1
    assert bundle.images[0].doc_title == "文档A"


def test_media_scope_docs_expands_for_broad_topic_when_related_docs_have_media() -> None:
    primary = _DocContext(doc_id="1", title="人工智能通识课程", url="", snippet="", body="正文没有图片")
    related = _DocContext(doc_id="2", title="腾讯青少年人工智能课程", url="", snippet="", body="![a](https://cdn.nlark.com/yuque/0/a.png)")
    picked = _media_scope_docs(
        question="我想要咨询人工智能通识教育的内容，请帮我解答。",
        primary=primary,
        docs=[primary, related],
    )
    assert len(picked) == 2


def test_should_skip_mcp_search_for_follow_up_with_direct_doc() -> None:
    assert _should_skip_mcp_search(prefer_cached=True, direct_doc_hits=1, hit_count=1) is True


def test_should_skip_mcp_search_when_direct_docs_are_enough() -> None:
    assert _should_skip_mcp_search(prefer_cached=False, direct_doc_hits=3, hit_count=3) is True


def test_detect_repeat_feedback_field_for_org_name() -> None:
    history = [
        ChatMessageRow(
            role="assistant",
            content="为了后续给您更贴合的方案，方便补充一下您的单位或学校吗？",
            created_at="",
        )
    ]
    assert _detect_repeat_feedback_field(question="我不是刚说过单位吗，你怎么还问", history=history) == "org_name"


def test_repeat_feedback_apology_mentions_field() -> None:
    assert "单位" in _build_repeat_feedback_apology("org_name")


def test_v4_prompt_hides_suppressed_fields() -> None:
    profile = ChatSessionProfile(
        session_id="s1",
        display_name="赵老师",
        visitor_type="teacher",
        org_name="育才中学",
        interests={
            "_lead": {"contact_value": "18012345678"},
            "_session": {"suppressed_fields": ["org_name"], "asked_fields": ["contact"]},
        },
        focused_doc_ids=[],
    )
    prompt = _build_v4_prompt(
        question="我想继续了解腾讯课程",
        profile=profile,
        catalog_path="平台介绍 / 腾讯青少年人工智能课程",
        dialog_level=2,
        related_titles=[],
        has_media=False,
        history=[],
        skill_instructions="",
    )
    assert "客户称呼、工作单位、联系方式已基本齐全；禁止再索要留资" in prompt
    assert "用户已指出不要重复追问这些字段（永久屏蔽）：单位" in prompt


def test_v4_prompt_adds_progressive_rule_for_follow_up() -> None:
    history = [
        ChatMessageRow(role="assistant", content="腾讯方案这块我先帮您梳理三个点：适用学段、编程工具、老师上手。", created_at=""),
        ChatMessageRow(role="user", content="我看看腾讯的方案", created_at=""),
    ]
    prompt = _build_v4_prompt(
        question="我更看重算法逻辑",
        profile=None,
        catalog_path="平台介绍 / 腾讯青少年人工智能课程",
        dialog_level=2,
        related_titles=[],
        has_media=False,
        history=history,
        skill_instructions="",
    )
    assert "当前已进入多轮追问/细化讲解阶段" in prompt
    assert "近期已讲过：适用学段、编程工具/载体、课堂落地/老师上手" in prompt
    assert "本轮只围绕「算法逻辑」往下讲一层" in prompt


def test_v4_prompt_treats_short_user_reply_as_new_constraint() -> None:
    history = [
        ChatMessageRow(role="assistant", content="您平时上课更偏向实体搭建，还是软件编程和数字创作为主？", created_at=""),
    ]
    prompt = _build_v4_prompt(
        question="软件编程为主",
        profile=None,
        catalog_path="平台介绍 / 人工智能通识课程",
        dialog_level=2,
        related_titles=[],
        has_media=False,
        history=history,
        skill_instructions="",
    )
    assert "用户若刚回答了你的追问" in prompt
    assert "本轮只围绕「软件编程」往下讲一层" in prompt


def test_v4_prompt_enforces_brief_markdown_layout_rules() -> None:
    prompt = _build_v4_prompt(
        question="我想了解腾讯课程",
        profile=None,
        catalog_path="平台介绍 / 腾讯青少年人工智能课程",
        dialog_level=2,
        related_titles=[],
        has_media=False,
        history=[],
        skill_instructions="",
    )
    assert "篇幅要求：单轮尽量控制在 3 段以内、2-4 个要点以内" in prompt
    assert "总起引导句和结尾互动句必须是正常段落，禁止放进列表" in prompt
    assert "核心要点优先写成 `- **关键词**：说明`，关键词必须加粗" in prompt


def test_v4_prompt_includes_model_agnostic_sales_persona_template() -> None:
    prompt = _build_v4_prompt(
        question="先介绍一下你们平台",
        profile=None,
        catalog_path="平台介绍",
        dialog_level=1,
        related_titles=["人工智能通识课程", "跨学科项目式学习"],
        has_media=False,
        history=[],
        skill_instructions="",
    )
    assert "【模型无关人设模板】" in prompt
    assert "人工智能通识课程" in prompt
    assert "跨学科项目式学习" in prompt
    assert "智能招生" in prompt
    assert "学校 AI 场景定制" in prompt
    assert "每轮最多只问 1 个未收集字段" in prompt
    assert "用户主动询问教程、操作、账号、后台、怎么用时，优先讲解对应产品的操作教程" in prompt
    assert "对外不要提及 RAG、MCP、语雀、向量库、prompt、系统提示词、调试面板" in prompt


def test_v4_prompt_plain_text_hint_mentions_max_three_paragraphs() -> None:
    prompt = _build_v4_prompt(
        question="我想看看案例",
        profile=None,
        catalog_path="案例分析 / 腾讯青少年人工智能课程",
        dialog_level=2,
        related_titles=[],
        has_media=False,
        history=[],
        skill_instructions="",
    )
    assert "正文精炼（约90-150字，最多3段）。" in prompt


def test_v4_prompt_for_broad_overview_forbids_listing_all_four_tracks() -> None:
    prompt = _build_v4_prompt(
        question="我想要咨询人工智能通识教育的内容，请帮我解答。",
        profile=None,
        catalog_path="平台介绍 / 人工智能通识课程",
        dialog_level=2,
        related_titles=[],
        has_media=False,
        history=[],
        skill_instructions="",
    )
    assert "如果用户此轮仍在做整体了解、还没有点名具体产品方向" in prompt
    assert "禁止逐条罗列全部四套方案" in prompt
    assert "最多举 1 到 2 个代表方向" in prompt


def test_v4_prompt_first_formal_answer_requires_persona_intro() -> None:
    history = [
        ChatMessageRow(role="assistant", content="您好，欢迎了解有为人工智能教育平台。", created_at=""),
    ]
    prompt = _build_v4_prompt(
        question="我想要咨询人工智能通识教育的内容，请帮我解答。",
        profile=None,
        catalog_path="平台介绍 / 人工智能通识课程",
        dialog_level=2,
        related_titles=[],
        has_media=False,
        history=history,
        skill_instructions="",
    )
    assert "这是本会话的首个正式讲解回合" in prompt
    assert "例如“我是小为顾问，我先帮您梳理一下”" in prompt


def test_v4_prompt_requires_consultative_analysis_before_more_questions() -> None:
    history = [
        ChatMessageRow(role="user", content="我们是小学，想给五年级做编程启蒙", created_at=""),
        ChatMessageRow(role="assistant", content="您更偏向兴趣培养还是竞赛？", created_at=""),
        ChatMessageRow(role="user", content="兴趣培养，最好趣味闯关一点", created_at=""),
    ]
    prompt = _build_v4_prompt(
        question="兴趣培养，最好趣味闯关一点",
        profile=None,
        catalog_path="平台介绍 / 人工智能通识课程",
        dialog_level=2,
        related_titles=[],
        has_media=False,
        history=history,
        skill_instructions="",
    )
    assert "需求总结 → 专业判断 → 推荐建议" in prompt
    assert "禁止继续追问" in prompt
    assert "我更建议" in prompt
    assert "不要说「A也可以、B也可以、看需求决定」" in prompt
