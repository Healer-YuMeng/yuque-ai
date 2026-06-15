from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.admin_api import router as admin_router
from app.api.chat_api import get_qa_service, router as chat_router
from app.db.admin_customers import AdminCustomerRepository
from app.db.profile_repository import ChatSessionProfileRepository
from app.db.repositories import ChatSessionRepository, DocumentRepository, LeadCaptureRepository, QALogRepository
from app.db.session import DatabaseSessionFactory
from app.service.qa_service import QAService


def _build_service(tmp_path: Path) -> QAService:
    session_factory = DatabaseSessionFactory(str(tmp_path / "visitor_trial.db"))
    document_repo = DocumentRepository(session_factory)
    service = QAService.__new__(QAService)
    service._lead_capture_repository = LeadCaptureRepository(session_factory)
    service._chat_session_profile_repository = ChatSessionProfileRepository(session_factory)
    service._document_repository = document_repo
    return service


def _build_client(tmp_path: Path) -> TestClient:
    app = FastAPI()
    service = _build_service(tmp_path)
    customer_repo = AdminCustomerRepository(DatabaseSessionFactory(str(tmp_path / "visitor_trial.db")))

    @app.on_event("startup")
    async def _startup() -> None:
        await service._document_repository.init_db()
        app.state.admin_customer_repository = customer_repo

    app.dependency_overrides[get_qa_service] = lambda: service
    app.include_router(chat_router)
    app.include_router(admin_router)
    return TestClient(app)


def test_visitor_trial_apply_persists_customer_for_admin(tmp_path: Path) -> None:
    client = _build_client(tmp_path)

    with client:
        resp = client.post(
            "/visitor/trial/apply",
            json={
                "session_id": "sess_visitor_apply_1",
                "name": "赵老师",
                "org_name": "培训机构",
                "contact": "18273648765",
                "interested_product": "人工智能通识教育",
            },
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["ok"] is True
        assert "提交成功" in payload["message"]

        list_resp = client.get("/admin-api/customers")
        assert list_resp.status_code == 200
        items = list_resp.json()["items"]
        assert len(items) == 1
        assert items[0]["display_name"] == "赵老师"
        assert items[0]["org_name"] == "培训机构"
        assert "18273648765" in items[0]["contact"]
        assert items[0]["follow_up_status"] == "待跟进"
        assert items[0]["trial_account"] == "待发放"


@pytest.mark.asyncio
async def test_apply_visitor_trial_account_sets_admin_defaults(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    await service._document_repository.init_db()

    result = await service.apply_visitor_trial_account(
        session_id="sess_defaults",
        name="李老师",
        org_name="实验小学",
        contact="13800138000",
    )

    assert result.ok is True
    profile = await service._chat_session_profile_repository.get_profile(session_id="sess_defaults")
    assert profile is not None
    assert profile.display_name == "李老师"
    assert profile.org_name == "实验小学"
    admin_meta = (profile.interests or {}).get("_admin") or {}
    assert admin_meta.get("follow_up_status") == "待跟进"
    assert admin_meta.get("test_account_status") == "待发放"
    session_meta = (profile.interests or {}).get("_session") or {}
    assert session_meta.get("trial_apply_submitted") is True
