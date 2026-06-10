from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from app.api.chat_api import get_qa_service, router
from app.conversation.lead_nudge_policy import LeadNudgePolicy
from app.db.profile_repository import ChatSessionProfileRepository
from app.db.repositories import ChatSessionRepository, DocumentRepository, LeadCaptureRepository
from app.db.session import DatabaseSessionFactory
from app.schemas.chat import ChatV4Response
from app.service import qa_service as qa_service_module
from app.service.qa_service import QAService


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


@pytest.mark.asyncio
async def test_chat_v4_stream_passes_prior_history_without_current_turn(tmp_path, monkeypatch) -> None:
    session_factory = DatabaseSessionFactory(str(tmp_path / "chat.db"))
    document_repository = DocumentRepository(session_factory)
    await document_repository.init_db()

    chat_repo = ChatSessionRepository(session_factory)
    service = QAService.__new__(QAService)
    service._chat_session_repository = chat_repo
    service._chat_session_profile_repository = ChatSessionProfileRepository(session_factory)
    service._lead_capture_repository = LeadCaptureRepository(session_factory)
    service._lead_nudge_policy = LeadNudgePolicy(rounds_threshold=3, stay_seconds_threshold=60)
    service._guide_toc_nodes = []
    service._refresh_guide_doc_titles_if_stale = lambda *args, **kwargs: _async_none()
    service._compute_yuque_scope = lambda owner, token_profile: None
    service._build_generator_by_selected_model = lambda model: object()
    service._build_mcp_client = lambda scope: object()

    captured: dict[str, object] = {}

    class _CapturingV4Orchestrator:
        def __init__(self, **kwargs):
            pass

        async def answer_stream(self, *, question, session_id, history, selected_doc_ids=()):
            captured["history"] = list(history)
            yield {
                "event": "done",
                "data": ChatV4Response(answer="ok", sources=[], fallback_used=False).model_dump(),
            }

    monkeypatch.setattr(qa_service_module, "SalesDialogOrchestratorV4", _CapturingV4Orchestrator)

    events = [
        item
        async for item in service.chat_v4_stream(
            "我想要咨询跨学科项目化学习的内容，请帮我解答。",
            chat_mode="visitor_sales",
            session_id="s-first-turn",
        )
    ]

    assert events[-1]["event"] == "done"
    assert captured["history"] == []


async def _async_none() -> None:
    return None
