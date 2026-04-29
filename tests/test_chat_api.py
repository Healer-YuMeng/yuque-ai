from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.chat_api import get_qa_service, router
from app.rag.generator import GeneratorConfigError
from app.schemas.chat import ChatResponse, SourceItem


class FakeQAService:
    async def chat(self, question: str, *, model=None, owner=None) -> ChatResponse:
        return ChatResponse(
            answer=f"回答: {question}",
            fallback_used=False,
            sources=[SourceItem(title="测试文档", url=None, source_type="vector")],
        )

    async def rebuild_index(self, *, bootstrap_query: str):
        return 2, 6

    def runtime_mode(self):
        return "direct_yuque", "语雀直连模式"


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
        async def chat(self, question: str, *, model=None, owner=None) -> ChatResponse:
            raise GeneratorConfigError("缺少 LLM API key。")

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_qa_service] = lambda: BrokenQAService()
    client = TestClient(app)

    response = client.post("/chat", json={"question": "退款多久到账？"})

    assert response.status_code == 503

