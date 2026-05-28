from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.chat import ChatMediaBundle, ChatRequest, ChatV2Response, MediaItem, SelectedYuqueDocRef, SourceItem


def test_chat_request_defaults_visitor_sales() -> None:
    body = ChatRequest(question="你好")
    assert body.chat_mode == "visitor_sales"
    assert body.session_id is None


def test_chat_request_accepts_session_id() -> None:
    body = ChatRequest(question="你好", session_id="s-abc-123", chat_mode="rag")
    assert body.session_id == "s-abc-123"
    assert body.chat_mode == "rag"


def test_chat_request_empty_owner_becomes_none() -> None:
    body = ChatRequest(question="你好", owner="")
    assert body.owner is None


def test_chat_request_selected_yuque_docs_roundtrip() -> None:
    body = ChatRequest(
        question="你好",
        selected_yuque_docs=[SelectedYuqueDocRef(doc_id=1001, slug="my-doc", title="标题")],
    )
    dumped = body.model_dump()
    assert dumped["selected_yuque_docs"][0]["doc_id"] == 1001
    assert dumped["selected_yuque_docs"][0]["slug"] == "my-doc"


def test_selected_yuque_doc_rejects_non_positive_id() -> None:
    with pytest.raises(ValidationError):
        SelectedYuqueDocRef(doc_id=0, title="x")


def test_chat_v2_response_media_defaults() -> None:
    body = ChatV2Response(answer="ok", sources=[SourceItem(title="x", source_type="mcp")])
    assert body.answer_style == "short_sales"
    assert body.media.images == []
    assert body.media.videos == []


def test_chat_v2_response_media_roundtrip() -> None:
    body = ChatV2Response(
        answer="ok",
        sources=[SourceItem(title="x", source_type="mcp")],
        media=ChatMediaBundle(
            images=[MediaItem(url="https://cdn.example.com/a.png", title="图")],
            videos=[MediaItem(url="https://cdn.example.com/a.mp4", title="视")],
        ),
    )
    dumped = body.model_dump()
    assert dumped["media"]["images"][0]["url"].endswith(".png")
    assert dumped["media"]["videos"][0]["url"].endswith(".mp4")
