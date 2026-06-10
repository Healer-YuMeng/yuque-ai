from __future__ import annotations

import pytest

from app.conversation.profile_extractor import ProfileExtractor
from app.db.profile_repository import ChatSessionProfile
from app.db.repositories import ChatMessageRow


@pytest.mark.asyncio
async def test_profile_extractor_rule_picks_name_org_role() -> None:
    ex = ProfileExtractor()
    history = [
        ChatMessageRow(role="user", content="你好，我是李老师，来自育才中学", created_at="2026-01-01 10:00:00"),
        ChatMessageRow(role="assistant", content="好的", created_at="2026-01-01 10:00:01"),
    ]
    prof = ChatSessionProfile(
        session_id="s1",
        display_name="",
        visitor_type="",
        org_name="",
        interests={},
        focused_doc_ids=[],
    )
    upd = await ex.extract_update(question="我想了解备课流程", history=history, current_profile=prof)
    assert upd.visitor_type == "teacher"
    assert upd.display_name == "李老师"


@pytest.mark.asyncio
async def test_profile_extractor_zhao_parent() -> None:
    ex = ProfileExtractor()
    upd = await ex.extract_update(question="你好，我是赵先生，是一名家长", history=[], current_profile=None)
    assert upd.visitor_type == "parent"
    assert upd.display_name and "赵" in upd.display_name


@pytest.mark.asyncio
async def test_profile_extractor_typo_shi_zhang() -> None:
    ex = ProfileExtractor()
    upd = await ex.extract_update(question="你好，我时张老师，来自育才中学", history=[], current_profile=None)
    assert upd.display_name and "张" in upd.display_name
    assert upd.org_name and "育才" in upd.org_name
    assert upd.org_name and "中学" in upd.org_name
    assert upd.interests and any(k for k in upd.interests.keys())


@pytest.mark.asyncio
async def test_profile_extractor_keeps_full_honorific_and_splits_org() -> None:
    ex = ProfileExtractor()
    upd = await ex.extract_update(question="你好，我是育才中学的赵老师", history=[], current_profile=None)
    assert upd.display_name == "赵老师"
    assert upd.org_name == "育才中学"
    assert upd.visitor_type == "teacher"


@pytest.mark.asyncio
async def test_profile_extractor_does_not_treat_grade_as_name() -> None:
    ex = ProfileExtractor()
    upd = await ex.extract_update(question="我时高年级", history=[], current_profile=None)
    assert not upd.display_name


@pytest.mark.asyncio
async def test_profile_extractor_does_not_treat_stage_phrase_as_name() -> None:
    ex = ProfileExtractor()
    upd = await ex.extract_update(question="我是给小学", history=[], current_profile=None)
    assert not upd.display_name


@pytest.mark.asyncio
async def test_profile_extractor_xing_zhao_becomes_teacher() -> None:
    ex = ProfileExtractor()
    upd = await ex.extract_update(question="我姓赵", history=[], current_profile=None)
    assert upd.display_name == "赵老师"


@pytest.mark.asyncio
async def test_profile_extractor_does_not_treat_org_role_phrase_as_name() -> None:
    ex = ProfileExtractor()
    upd = await ex.extract_update(question="我是在育才中学做老师", history=[], current_profile=None)
    assert not upd.display_name
    assert upd.org_name == "育才中学"
    assert upd.visitor_type == "teacher"
