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
from app.conversation.user_info_extractor import StructuredUserInfo
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
    client = TestClient(app)
    client.post("/admin-api/auth/login", json={"username": "admin", "password": "admin123456"})
    return client


class _FakeStructuredUserInfoExtractor:
    def __init__(self) -> None:
        self.seen_transcript = ""

    async def extract(self, transcript: str) -> StructuredUserInfo:
        self.seen_transcript = transcript
        return StructuredUserInfo(
            display_name="zjt",
            org_name="xx学校",
            contact="13813655304",
            email="zjt@test.com",
        )


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
                "email": "zhao@example.com",
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

def test_visitor_trial_apply_accepts_email_without_contact(tmp_path: Path) -> None:
    client = _build_client(tmp_path)

    with client:
        resp = client.post(
            "/visitor/trial/apply",
            json={
                "session_id": "sess_visitor_email_apply",
                "name": "双老师",
                "org_name": "有为教育小学",
                "contact": "",
                "email": "yumeng@mc2.cn",
                "interested_product": "智能招生",
            },
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["ok"] is True

        list_resp = client.get("/admin-api/customers")
        assert list_resp.status_code == 200
        items = list_resp.json()["items"]
        assert len(items) == 1
        assert items[0]["display_name"] == "双老师"
        assert items[0]["org_name"] == "有为教育小学"
        assert items[0]["contact"] == ""
        assert items[0]["email"] == "yumeng@mc2.cn"

@pytest.mark.asyncio
async def test_apply_visitor_trial_account_sets_admin_defaults(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    await service._document_repository.init_db()

    result = await service.apply_visitor_trial_account(
        session_id="sess_defaults",
        name="李老师",
        org_name="实验小学",
        contact="13800138000",
        email="li@example.com",
    )

    assert result.ok is True
    profile = await service._chat_session_profile_repository.get_profile(session_id="sess_defaults")
    assert profile is not None
    assert profile.display_name == "李老师"
    assert profile.org_name == "实验小学"
    admin_meta = (profile.interests or {}).get("_admin") or {}
    assert admin_meta.get("follow_up_status") == "待跟进"
    assert admin_meta.get("test_account_status") == "待发放"
    lead_meta = (profile.interests or {}).get("_lead") or {}
    assert lead_meta.get("email") == "li@example.com"
    session_meta = (profile.interests or {}).get("_session") or {}
    assert session_meta.get("trial_apply_submitted") is True


@pytest.mark.asyncio
async def test_apply_visitor_trial_account_cleans_user_info_before_persisting(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    await service._document_repository.init_db()

    result = await service.apply_visitor_trial_account(
        session_id="sess_clean_apply",
        name="我的名字是 zjt",
        org_name="是xx",
        contact="手机号 13813655304",
        email="邮箱是 ZJT@Test.COM",
    )

    assert result.ok is True
    profile = await service._chat_session_profile_repository.get_profile(session_id="sess_clean_apply")
    assert profile is not None
    assert profile.display_name == "zjt"
    assert profile.org_name == "xx"
    lead_meta = (profile.interests or {}).get("_lead") or {}
    assert lead_meta.get("name") == "zjt"
    assert lead_meta.get("org_name") == "xx"
    assert lead_meta.get("contact_value") == "13813655304"
    assert lead_meta.get("email") == "zjt@test.com"


@pytest.mark.asyncio
async def test_apply_visitor_trial_account_uses_structured_extractor_before_persisting(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    extractor = _FakeStructuredUserInfoExtractor()
    service._user_info_extractor = extractor
    await service._document_repository.init_db()

    result = await service.apply_visitor_trial_account(
        session_id="sess_structured_apply",
        name="我叫 zjt，不要存整句",
        org_name="单位是 xx学校",
        contact="联系方式是 13813655304",
        email="邮箱是 zjt@test.com",
    )

    assert result.ok is True
    assert "姓名：我叫 zjt，不要存整句" in extractor.seen_transcript
    profile = await service._chat_session_profile_repository.get_profile(session_id="sess_structured_apply")
    assert profile is not None
    assert profile.display_name == "zjt"
    assert profile.org_name == "xx学校"
    lead_meta = (profile.interests or {}).get("_lead") or {}
    assert lead_meta.get("contact_value") == "13813655304"
    assert lead_meta.get("email") == "zjt@test.com"


@pytest.mark.asyncio
async def test_v5_chat_contact_persists_customer_for_admin(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    await service._document_repository.init_db()
    await service._chat_session_profile_repository.upsert_profile(
        session_id="sess_v5_chat_lead",
        display_name="赵老师",
        org_name="",
        interests={
            "_lead": {
                "contact_type": "phone",
                "contact_value": "13423445679",
                "interested_product": "智能招生",
            }
        },
    )

    saved = await service._persist_v5_chat_lead_for_admin(
        session_id="sess_v5_chat_lead",
        question="我是赵老师，联系方式是13423445679",
        scene="智能招生",
    )

    assert saved is True
    customer_repo = AdminCustomerRepository(DatabaseSessionFactory(str(tmp_path / "visitor_trial.db")))
    items, total = await customer_repo.list_customers()
    assert total == 1
    assert items[0].display_name == "赵老师"
    assert items[0].contact == "13423445679"
    assert items[0].follow_up_status == "待跟进"
    assert items[0].trial_account == "待发放"


@pytest.mark.asyncio
async def test_v5_chat_email_only_persists_customer_for_admin(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    await service._document_repository.init_db()
    await service._chat_session_profile_repository.upsert_profile(
        session_id="sess_v5_chat_email_lead",
        display_name="王校长",
        org_name="有为中学",
        visitor_type="institution_decision_maker",
        interests={"_lead": {"interested_product": "人工智能通识课程"}},
    )

    saved = await service._persist_v5_chat_lead_for_admin(
        session_id="sess_v5_chat_email_lead",
        question="我的邮箱是yument@test.com，后续可以发资料到这里",
        scene="人工智能通识教育",
    )

    assert saved is True
    customer_repo = AdminCustomerRepository(DatabaseSessionFactory(str(tmp_path / "visitor_trial.db")))
    items, total = await customer_repo.list_customers()
    assert total == 1
    assert items[0].display_name == "王校长"
    assert items[0].contact == ""
    assert items[0].email == "yument@test.com"
    assert items[0].role_category == "机构/学校负责人"
