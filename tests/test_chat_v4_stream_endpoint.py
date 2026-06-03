from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.chat_api import get_qa_service, router
from app.schemas.chat import ChatV4Response


class _FakeQAServiceV4:
    async def chat_v4_stream(
        self,
        question: str,
        *,
        model=None,
        owner=None,
        token_profile=None,
        chat_mode=None,
        session_id=None,
        selected_yuque_docs=None,
    ):
        yield {"event": "token", "data": {"token": "你好"}}
        yield {
            "event": "done",
            "data": ChatV4Response(
                answer="你好",
                sources=[],
                fallback_used=True,
                debug={
                    "mode": "v4_content",
                    "turn_trace": {
                        "pipeline": "v4_content",
                        "mcp_calls": [{"tool": "yuque_search", "query": "测试", "hit_count": 1}],
                        "skills": [{"skill_id": "smart-summary", "reason": "测试"}],
                        "documents": [],
                    },
                },
            ).model_dump(),
        }

    def guide_titles_state(self):
        return {"total_nodes": 10}


def build_test_app() -> FastAPI:
    test_app = FastAPI()
    test_app.include_router(router)
    test_app.dependency_overrides[get_qa_service] = lambda: _FakeQAServiceV4()
    return test_app


def _set_chat_v4_enabled(value: bool) -> bool:
    from app.api import chat_api as chat_api_module

    old = bool(chat_api_module.settings.chat_v4_enabled)
    object.__setattr__(chat_api_module.settings, "chat_v4_enabled", value)
    return old


def test_chat_v4_stream_endpoint_enabled() -> None:
    old = _set_chat_v4_enabled(True)
    try:
        client = TestClient(build_test_app())
        resp = client.post("/chat/v4/stream", json={"question": "测试"})
        assert resp.status_code == 200
        assert "event: done" in resp.text
        assert "turn_trace" in resp.text
        assert "smart-summary" in resp.text
    finally:
        from app.api import chat_api as chat_api_module

        object.__setattr__(chat_api_module.settings, "chat_v4_enabled", old)


def test_chat_v4_capabilities_endpoint() -> None:
    old = _set_chat_v4_enabled(True)
    try:
        client = TestClient(build_test_app())
        resp = client.get("/chat/v4/capabilities")
        assert resp.status_code == 200
        assert resp.json()["enabled"] is True
    finally:
        from app.api import chat_api as chat_api_module

        object.__setattr__(chat_api_module.settings, "chat_v4_enabled", old)
