from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.chat_api import get_qa_service
from app.api.chat_v5_api import router
from app.schemas.chat_v5 import ChatV5DonePayload, FriendV5SourceItem


class _FakeQAServiceV5:
    async def chat_v5_stream(
        self,
        question: str,
        *,
        model=None,
        owner=None,
        token_profile=None,
        chat_mode=None,
        session_id=None,
        scene=None,
        trigger_type=None,
    ):
        yield {"event": "stage", "data": {"stage": "searching", "detail": "小为正在结合联网搜索梳理资料..."}}
        yield {"event": "token", "data": {"token": "我是小为。"}}
        yield {
            "event": "done",
            "data": ChatV5DonePayload(
                answer="我是小为。",
                tags=["想看课程例子？", "想了解适合年级？", "想看看落地方式？"],
                sources=[
                    FriendV5SourceItem(
                        source_type="web",
                        title="AI 教育报道",
                        url="https://example.com/web",
                        snippet="联网搜索摘要",
                        index=1,
                    )
                ],
                search_keywords=["人工智能通识教育", "AI 教育"],
                profile_fields={},
            ).model_dump(),
        }

    def chat_v5_capabilities(self):
        return {"enabled": True, "model": "qwen3.7-plus", "require_web_sources": True}


def build_test_app() -> FastAPI:
    test_app = FastAPI()
    test_app.include_router(router)
    test_app.dependency_overrides[get_qa_service] = lambda: _FakeQAServiceV5()
    return test_app


def _set_chat_v5_enabled(value: bool) -> bool:
    from app.api import chat_v5_api as chat_v5_api_module

    old = bool(chat_v5_api_module.settings.chat_v5_enabled)
    object.__setattr__(chat_v5_api_module.settings, "chat_v5_enabled", value)
    return old


def test_chat_v5_stream_disabled_returns_sse_error() -> None:
    old = _set_chat_v5_enabled(False)
    try:
        client = TestClient(build_test_app())
        resp = client.post(
            "/chat/v5/stream",
            json={
                "chat_mode": "friend_v5",
                "question": "人工智能通识教育",
                "session_id": "sess_v5_disabled",
                "scene": "人工智能通识教育",
                "trigger_type": "scene",
            },
        )

        assert resp.status_code == 200
        assert "event: error" in resp.text
        assert "V5 链路未开启" in resp.text
    finally:
        from app.api import chat_v5_api as chat_v5_api_module

        object.__setattr__(chat_v5_api_module.settings, "chat_v5_enabled", old)


def test_chat_v5_stream_enabled_returns_token_done_and_web_source() -> None:
    old = _set_chat_v5_enabled(True)
    try:
        client = TestClient(build_test_app())
        resp = client.post(
            "/chat/v5/stream",
            json={
                "chat_mode": "friend_v5",
                "question": "人工智能通识教育",
                "session_id": "sess_v5_ok",
                "scene": "人工智能通识教育",
                "trigger_type": "scene",
                "model": "qwen3.7-plus",
            },
        )

        assert resp.status_code == 200
        assert "event: token" in resp.text
        assert "event: done" in resp.text
        assert '"source_type":"web"' in resp.text
        assert '"search_keywords":["人工智能通识教育","AI 教育"]' in resp.text
    finally:
        from app.api import chat_v5_api as chat_v5_api_module

        object.__setattr__(chat_v5_api_module.settings, "chat_v5_enabled", old)
