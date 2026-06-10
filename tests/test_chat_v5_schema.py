from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.chat import ResetSessionRequest
from app.schemas.chat_v5 import ChatV5DonePayload, ChatV5Request, FriendV5SourceItem


def test_chat_v5_request_accepts_required_fields() -> None:
    req = ChatV5Request(
        question="人工智能通识教育",
        session_id="sess_v5_schema_ok",
        scene="人工智能通识教育",
        trigger_type="scene",
    )

    assert req.chat_mode == "friend_v5"
    assert req.model is None


def test_chat_v5_request_requires_session_prefix() -> None:
    with pytest.raises(ValidationError):
        ChatV5Request(
            question="人工智能通识教育",
            session_id="sess_v4_wrong",
            scene="人工智能通识教育",
            trigger_type="scene",
        )


def test_chat_v5_request_requires_trigger_type() -> None:
    with pytest.raises(ValidationError):
        ChatV5Request(
            question="人工智能通识教育",
            session_id="sess_v5_missing_trigger",
            scene="人工智能通识教育",
        )


def test_chat_v5_request_rejects_invalid_scene() -> None:
    with pytest.raises(ValidationError):
        ChatV5Request(
            question="随便看看",
            session_id="sess_v5_bad_scene",
            scene="未知场景",
            trigger_type="manual",
        )


def test_reset_session_accepts_friend_v5() -> None:
    req = ResetSessionRequest(session_id="sess_v5_reset", chat_mode="friend_v5")

    assert req.chat_mode == "friend_v5"


def test_chat_v5_done_payload_accepts_web_and_yuque_sources() -> None:
    payload = ChatV5DonePayload(
        answer="小为帮你整理好了。",
        tags=["想看课程例子？", "想了解适合年级？", "想看看落地方式？"],
        search_keywords=["上海有为云科技有限公司", "人工智能通识教育产品"],
        sources=[
            FriendV5SourceItem(
                source_type="web",
                title="人工智能教育报道",
                url="https://example.com/web",
                snippet="联网搜索片段",
                index=1,
            ),
            FriendV5SourceItem(
                source_type="yuque",
                title="乐高人工智能课程介绍",
                url="https://www.yuque.com/example/doc",
                doc_id="123",
            ),
        ],
        profile_fields={"display_name": "赵老师", "org_name": "第一中学"},
    )

    assert [item.source_type for item in payload.sources] == ["web", "yuque"]
    assert payload.search_keywords == ["上海有为云科技有限公司", "人工智能通识教育产品"]
