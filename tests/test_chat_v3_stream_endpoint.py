import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.chat_api import get_qa_service, router
from app.schemas.chat import ChatV3Response


class _FakeQAService:
    async def chat_v3_stream(
        self,
        question: str,
        *,
        model=None,
        owner=None,
        token_profile=None,
        chat_mode=None,
        session_id=None,
    ):
        yield {"event": "stage", "data": {"stage": "retrieving", "detail": "v3", "mode": "v3"}}
        yield {"event": "token", "data": {"token": "你"}}
        yield {"event": "token", "data": {"token": "好"}}
        yield {"event": "done", "data": ChatV3Response(answer="你好", sources=[], fallback_used=True).model_dump()}

    def guide_titles_state(self):
        return {"total_nodes": 1}


def test_v3_selected_title_triggers_answer_path() -> None:
    # 这里只验证路由可用；编排器细节在 service 单测覆盖
    old = _set_chat_v3_enabled(True)
    try:
        client = TestClient(build_test_app())
        resp = client.post("/chat/v3/stream", json={"question": "我想了解跨学科项目式学习"})
        assert resp.status_code == 200
        assert "event: done" in resp.text
    finally:
        from app.api import chat_api as chat_api_module

        object.__setattr__(chat_api_module.settings, "chat_v3_enabled", old)


def _set_chat_v3_enabled(value: bool) -> bool:
    from app.api import chat_api as chat_api_module

    old = bool(chat_api_module.settings.chat_v3_enabled)
    object.__setattr__(chat_api_module.settings, "chat_v3_enabled", value)
    return old


def build_test_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_qa_service] = lambda: _FakeQAService()
    return app


def test_chat_v3_stream_endpoint_enabled() -> None:
    from app.api import chat_api as chat_api_module

    old = _set_chat_v3_enabled(True)
    try:
        client = TestClient(build_test_app())
        resp = client.post("/chat/v3/stream", json={"question": "hi"})
        assert resp.status_code == 200
        assert "event: token" in resp.text
        assert "event: done" in resp.text
    finally:
        object.__setattr__(chat_api_module.settings, "chat_v3_enabled", old)


def test_chat_v3_capabilities_endpoint() -> None:
    old = _set_chat_v3_enabled(True)
    try:
        client = TestClient(build_test_app())
        resp = client.get("/chat/v3/capabilities")
        assert resp.status_code == 200
        assert resp.json()["enabled"] is True
    finally:
        from app.api import chat_api as chat_api_module

        object.__setattr__(chat_api_module.settings, "chat_v3_enabled", old)

