import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from openai import APIConnectionError, APIStatusError

from app.api.chat_api import get_qa_service, router
from app.rag.generator import GeneratorConfigError
from app.schemas.chat import ChatMediaBundle, ChatResponse, ChatV2Response, MediaItem, SourceItem


class FakeQAService:
    def __init__(self) -> None:
        self.refresh_calls: list[bool] = []

    async def chat(
        self,
        question: str,
        *,
        model=None,
        owner=None,
        selected_yuque_docs=None,
        token_profile=None,
        chat_mode=None,
        session_id=None,
    ) -> ChatResponse:
        return ChatResponse(
            answer=f"回答: {question}",
            fallback_used=False,
            sources=[SourceItem(title="测试文档", url=None, source_type="vector")],
        )

    async def rebuild_index(self, *, bootstrap_query: str):
        return 2, 6

    def runtime_mode(self):
        return "direct_yuque", "语雀直连模式"

    async def chat_stream(
        self,
        question: str,
        *,
        model=None,
        owner=None,
        selected_yuque_docs=None,
        token_profile=None,
        chat_mode=None,
        session_id=None,
    ):
        yield {"event": "done", "data": {"answer": "ok", "sources": [], "fallback_used": False, "debug": {}}}

    async def list_session_messages(self, *, session_id: str, limit: int):
        class _Row:
            def __init__(self, role: str, content: str, created_at: str) -> None:
                self.role = role
                self.content = content
                self.created_at = created_at

        if session_id == "s1":
            return [
                _Row("user", "你好", "2026-01-01 00:00:00"),
                _Row("assistant", "你好，有什么想了解的？", "2026-01-01 00:00:01"),
            ][:limit]
        return []

    async def chat_v2(
        self,
        question: str,
        *,
        model=None,
        owner=None,
        token_profile=None,
        chat_mode=None,
        session_id=None,
    ) -> ChatV2Response:
        return ChatV2Response(
            answer=f"V2回答: {question}",
            fallback_used=False,
            sources=[SourceItem(title="测试文档V2", url=None, source_type="mcp")],
            media=ChatMediaBundle(
                images=[MediaItem(url="https://cdn.example.com/a.png", title="示例图")],
                videos=[MediaItem(url="https://cdn.example.com/demo.mp4", title="示例视频")],
            ),
            lead_nudge_triggered=True,
        )

    async def chat_v2_stream(
        self,
        question: str,
        *,
        model=None,
        owner=None,
        token_profile=None,
        chat_mode=None,
        session_id=None,
    ):
        yield {"event": "stage", "data": {"stage": "retrieving", "detail": "v2"}}
        yield {"event": "token", "data": {"token": "V2"}}
        yield {
            "event": "done",
            "data": {
                "answer": f"V2回答: {question}",
                "sources": [],
                "fallback_used": False,
                "answer_style": "short_sales",
                "media": {"images": [], "videos": []},
                "lead_nudge_triggered": False,
                "debug": {},
            },
        }

    def guide_titles_state(self):
        return {
            "v15_enabled": True,
            "count": 3,
            "titles": ["平台介绍", "使用指南", "案例与社区"],
            "total_nodes": 5,
            "root_nodes": 2,
            "max_level": 3,
            "nodes": [
                {
                    "uuid": "n1",
                    "title": "平台介绍",
                    "level": 1,
                    "parent_uuid": "",
                    "node_type": "title",
                    "url": "https://www.yuque.com/a/b/c1",
                    "doc_id": None,
                    "children": [
                        {
                            "uuid": "n1-1",
                            "title": "课程产品矩阵",
                            "level": 2,
                            "parent_uuid": "n1",
                            "node_type": "doc",
                            "url": "https://www.yuque.com/a/b/c2",
                            "doc_id": 101,
                            "children": [],
                        }
                    ],
                }
            ],
            "refresh_interval_s": 300,
            "refreshed_seconds_ago": 12.0,
        }

    async def refresh_guide_titles(self, *, force: bool = False) -> None:
        self.refresh_calls.append(force)


def build_test_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_qa_service] = lambda: FakeQAService()
    return app


def test_health_endpoint() -> None:
    client = TestClient(build_test_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_yuque_asset_rejects_disallowed_host() -> None:
    from app.data.yuque_images import encode_image_proxy_token

    client = TestClient(build_test_app())
    bad = encode_image_proxy_token("https://evil.example.com/a.png")
    r = client.get("/yuque/asset", params={"t": bad})
    assert r.status_code == 400


def test_chat_endpoint() -> None:
    client = TestClient(build_test_app())

    response = client.post("/chat", json={"question": "退款多久到账？"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "回答: 退款多久到账？"
    assert payload["sources"][0]["title"] == "测试文档"


def _set_chat_v15_enabled(value: bool) -> bool:
    from app.api import chat_api as chat_api_module

    old = bool(chat_api_module.settings.chat_v15_enabled)
    object.__setattr__(chat_api_module.settings, "chat_v15_enabled", value)
    return old


def test_chat_v2_endpoint_disabled_returns_503() -> None:
    from app.api import chat_api as chat_api_module

    old = _set_chat_v15_enabled(False)
    try:
        client = TestClient(build_test_app())
        response = client.post("/chat/v2", json={"question": "可以看视频吗？"})
        assert response.status_code == 503
    finally:
        object.__setattr__(chat_api_module.settings, "chat_v15_enabled", old)


def test_chat_v2_endpoint_enabled() -> None:
    from app.api import chat_api as chat_api_module

    old = _set_chat_v15_enabled(True)
    try:
        client = TestClient(build_test_app())
        response = client.post("/chat/v2", json={"question": "可以看视频吗？"})
        assert response.status_code == 200
        payload = response.json()
        assert payload["answer"].startswith("V2回答:")
        assert payload["media"]["videos"][0]["url"].endswith(".mp4")
    finally:
        object.__setattr__(chat_api_module.settings, "chat_v15_enabled", old)

def test_runtime_mode_endpoint() -> None:
    client = TestClient(build_test_app())

    response = client.get("/runtime-mode")

    assert response.status_code == 200
    assert response.json()["mode"] == "direct_yuque"


def test_chat_endpoint_returns_503_for_missing_llm_key() -> None:
    class BrokenQAService(FakeQAService):
        async def chat(
            self,
            question: str,
            *,
            model=None,
            owner=None,
            selected_yuque_docs=None,
            token_profile=None,
            chat_mode=None,
            session_id=None,
        ) -> ChatResponse:
            raise GeneratorConfigError("缺少 LLM API key。")

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_qa_service] = lambda: BrokenQAService()
    client = TestClient(app)

    response = client.post("/chat", json={"question": "退款多久到账？"})

    assert response.status_code == 503


def test_chat_stream_returns_sse_error_on_llm_connection_failure() -> None:
    class StreamConnFail(FakeQAService):
        async def chat_stream(
            self,
            question: str,
            *,
            model=None,
            owner=None,
            selected_yuque_docs=None,
            token_profile=None,
            chat_mode=None,
            session_id=None,
        ):
            yield {"event": "stage", "data": {"stage": "generating", "detail": "…"}}
            raise APIConnectionError(request=httpx.Request("POST", "https://api.example/v1"))

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_qa_service] = lambda: StreamConnFail()
    client = TestClient(app)

    response = client.post("/chat/stream", json={"question": "测试"})

    assert response.status_code == 200
    assert "event: error" in response.text
    assert "无法连接大模型服务" in response.text


def test_chat_stream_returns_friendly_sse_error_on_llm_503() -> None:
    req = httpx.Request("POST", "https://api.deepseek.com/v1/chat/completions")
    resp = httpx.Response(
        503,
        request=req,
        json={
            "error": {
                "message": "Service is too busy. We advise users to temporarily switch.",
                "type": "service_unavailable_error",
                "code": "service_unavailable_error",
            }
        },
    )

    class Stream503(FakeQAService):
        async def chat_stream(
            self,
            question: str,
            *,
            model=None,
            owner=None,
            selected_yuque_docs=None,
            token_profile=None,
            chat_mode=None,
            session_id=None,
        ):
            yield {"event": "stage", "data": {"stage": "generating", "detail": "…"}}
            raise APIStatusError("Service is too busy.", response=resp, body=resp.json())

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_qa_service] = lambda: Stream503()
    client = TestClient(app)

    response = client.post("/chat/stream", json={"question": "测试"})

    assert response.status_code == 200
    assert "event: error" in response.text
    assert "当前大模型服务端繁忙" in response.text
    assert "（503）" in response.text


def test_chat_v2_stream_endpoint_enabled() -> None:
    from app.api import chat_api as chat_api_module

    old = _set_chat_v15_enabled(True)
    try:
        client = TestClient(build_test_app())
        response = client.post("/chat/v2/stream", json={"question": "来个示例"})
        assert response.status_code == 200
        assert "event: done" in response.text
    finally:
        object.__setattr__(chat_api_module.settings, "chat_v15_enabled", old)


def test_chat_v2_guide_titles_endpoint() -> None:
    client = TestClient(build_test_app())
    response = client.get("/chat/v2/guide-titles")
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 3
    assert payload["total_nodes"] == 5
    assert payload["root_nodes"] == 2
    assert payload["max_level"] == 3
    assert payload["titles"][0] == "平台介绍"
    assert payload["nodes"][0]["children"][0]["title"] == "课程产品矩阵"


def test_chat_v2_guide_titles_refreshes_when_requested() -> None:
    fake = FakeQAService()
    test_app = build_test_app()
    test_app.dependency_overrides[get_qa_service] = lambda: fake

    client = TestClient(test_app)
    response = client.get("/chat/v2/guide-titles?refresh=true")

    assert response.status_code == 200
    assert fake.refresh_calls == [True]
