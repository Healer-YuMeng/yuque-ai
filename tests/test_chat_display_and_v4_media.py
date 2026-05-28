from __future__ import annotations

import pytest

from app.conversation.chat_display import display_name_for_chat
from app.conversation.profile_extractor import ProfileExtractor
from app.conversation.toc_catalog import CatalogNode
from app.conversation.v4_lead_outreach import _build_v4_nudge_text
from app.db.profile_repository import ChatSessionProfile
from app.service.media_answer_orchestrator import _DocContext, collect_media_from_doc_contexts
from app.service.sales_dialog_orchestrator_v4 import _should_show_media


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


def test_v4_nudge_lists_four_lead_fields() -> None:
    text = _build_v4_nudge_text(profile=None, lead_meta={})
    assert "姓名" in text
    assert "单位" in text
    assert "联系方式" in text
    assert "感兴趣的内容" in text


def test_v4_nudge_uses_full_display_name() -> None:
    profile = ChatSessionProfile(
        session_id="s1",
        display_name="赵",
        visitor_type="teacher",
        org_name="",
        interests={},
        focused_doc_ids=[],
    )
    text = _build_v4_nudge_text(profile=profile, lead_meta={})
    assert text.startswith("赵老师，")


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
