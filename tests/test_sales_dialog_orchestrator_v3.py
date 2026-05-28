from __future__ import annotations

from app.schemas.chat import GuideDocTitleNode
from app.service import sales_dialog_orchestrator_v3 as v3


def test_match_selected_title_fuzzy_tongshi() -> None:
    toc = [
        GuideDocTitleNode(
            uuid="1",
            title="人工智能通识课程",
            level=1,
            children=[],
        ),
        GuideDocTitleNode(uuid="2", title="平台介绍", level=1, children=[]),
    ]
    got = v3._match_selected_title(question="我想看看人工智能通识课", toc_tree=toc)
    assert got == "人工智能通识课程"


def test_match_selected_title_platform_intro() -> None:
    toc = [GuideDocTitleNode(uuid="2", title="平台介绍", level=1, children=[])]
    got = v3._match_selected_title(question="我想看平台介绍", toc_tree=toc)
    assert got == "平台介绍"


def test_guide_message_uses_name_not_org() -> None:
    from app.db.profile_repository import ChatSessionProfile

    prof = ChatSessionProfile(
        session_id="s",
        display_name="张老师",
        visitor_type="teacher",
        org_name="育才中学",
        interests={},
        focused_doc_ids=[],
    )

    class _Pick:
        title = "平台介绍"
        reason = ""

    msg = v3._build_guide_message(
        question="你好，我时张老师，来自育才中学",
        profile=prof,
        picks=[_Pick()],
        history=[],
    )
    assert "张老师" in msg
    assert "育才中学" not in msg
    assert "语雀" not in msg
    assert "知识库" not in msg
