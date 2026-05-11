import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from openai import APIConnectionError, APIStatusError

from app.api.chat_api import get_qa_service, router
from app.rag.generator import GeneratorConfigError
from app.schemas.chat import ChatResponse, SourceItem


class FakeQAService:
    async def chat(self, question: str, *, model=None, owner=None, selected_yuque_docs=None, token_profile=None) -> ChatResponse:
        return ChatResponse(
            answer=f"回答: {question}",
            fallback_used=False,
            sources=[SourceItem(title="测试文档", url=None, source_type="vector")],
        )

    async def rebuild_index(self, *, bootstrap_query: str):
        return 2, 6

    def runtime_mode(self):
        return "direct_yuque", "语雀直连模式"

    async def chat_stream(self, question: str, *, model=None, owner=None, selected_yuque_docs=None, token_profile=None):
        yield {"event": "done", "data": {"answer": "ok", "sources": [], "fallback_used": False, "debug": {}}}


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


def test_runtime_mode_endpoint() -> None:
    client = TestClient(build_test_app())

    response = client.get("/runtime-mode")

    assert response.status_code == 200
    assert response.json()["mode"] == "direct_yuque"


def test_chat_endpoint_returns_503_for_missing_llm_key() -> None:
    class BrokenQAService(FakeQAService):
        async def chat(self, question: str, *, model=None, owner=None, selected_yuque_docs=None, token_profile=None) -> ChatResponse:
            raise GeneratorConfigError("缺少 LLM API key。")

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_qa_service] = lambda: BrokenQAService()
    client = TestClient(app)

    response = client.post("/chat", json={"question": "退款多久到账？"})

    assert response.status_code == 503


def test_chat_stream_returns_sse_error_on_llm_connection_failure() -> None:
    class StreamConnFail(FakeQAService):
        async def chat_stream(self, question: str, *, model=None, owner=None, selected_yuque_docs=None, token_profile=None):
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
        async def chat_stream(self, question: str, *, model=None, owner=None, selected_yuque_docs=None, token_profile=None):
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

