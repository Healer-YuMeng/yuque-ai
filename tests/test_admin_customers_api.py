from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.admin_api import router
from app.db.admin_customers import AdminCustomerRepository
from app.db.repositories import DocumentRepository
from app.db.session import DatabaseSessionFactory


def build_admin_customer_test_client(tmp_path: Path) -> TestClient:
    app = FastAPI()
    db_path = tmp_path / "admin_customers.db"
    session_factory = DatabaseSessionFactory(str(db_path))
    document_repo = DocumentRepository(session_factory)
    customer_repo = AdminCustomerRepository(session_factory)

    @app.on_event("startup")
    async def _startup() -> None:
        await document_repo.init_db()
        app.state.admin_customer_repository = customer_repo

    app.include_router(router)
    client = TestClient(app)
    client.post("/admin-api/auth/login", json={"username": "admin", "password": "admin123456"})
    return client


def _trial_apply_interests(*, trial_issued: bool = True, username: str = "") -> dict:
    interests: dict = {
        "_lead": {"wants_trial": True, "contact_value": "13800138000", "contact_type": "phone"},
        "_session": {"trial_apply_submitted": True, "trial_account_issued": trial_issued},
    }
    if username:
        interests["_trial"] = {"username": username}
    return interests


async def _seed_trial_customer(
    session_factory: DatabaseSessionFactory,
    *,
    session_id: str,
    display_name: str,
    org_name: str,
    contact: str = "13800138000",
    email: str = "",
    visitor_type: str = "",
) -> None:
    conn = await session_factory.connect()
    try:
        await conn.execute(
            "INSERT INTO chat_session_profiles(session_id, display_name, org_name, visitor_type, interests_json) VALUES (?, ?, ?, ?, ?)",
            (
                session_id,
                display_name,
                org_name,
                visitor_type,
                json.dumps(
                    {
                        **_trial_apply_interests(),
                        "_lead": {
                            **_trial_apply_interests()["_lead"],
                            "contact_value": contact,
                            "email": email,
                        },
                    },
                    ensure_ascii=False,
                ),
            ),
        )
        await conn.execute(
            "INSERT INTO lead_captures(session_id, contact_type, contact_value, visitor_type) VALUES (?, ?, ?, ?)",
            (session_id, "phone", contact, visitor_type or None),
        )
        await conn.commit()
    finally:
        await conn.close()


def test_list_customers_only_shows_trial_apply_submissions(tmp_path: Path) -> None:
    client = build_admin_customer_test_client(tmp_path)
    session_factory = DatabaseSessionFactory(str(tmp_path / "admin_customers.db"))

    with client:
        asyncio.run(_seed_trial_customer(session_factory, session_id="sess_apply_1", display_name="张老师", org_name="实验小学"))

        async def seed_non_apply() -> None:
            conn = await session_factory.connect()
            try:
                await conn.execute(
                    "INSERT INTO chat_session_profiles(session_id, display_name, org_name, interests_json) VALUES (?, ?, ?, ?)",
                    ("sess_chat_only", "路人甲", "某单位", json.dumps({"_lead": {"wants_trial": True}}, ensure_ascii=False)),
                )
                await conn.commit()
            finally:
                await conn.close()

        asyncio.run(seed_non_apply())

        list_resp = client.get("/admin-api/customers")
        assert list_resp.status_code == 200
        payload = list_resp.json()
        assert payload["total"] == 1
        assert len(payload["items"]) == 1
        assert payload["items"][0]["display_name"] == "张老师"
        assert payload["items"][0]["role_category"] == ""
        assert payload["page"] == 1
        assert payload["page_size"] == 10


def test_list_customers_includes_leads_even_without_trial_apply(tmp_path: Path) -> None:
    client = build_admin_customer_test_client(tmp_path)
    session_factory = DatabaseSessionFactory(str(tmp_path / "admin_customers.db"))

    with client:
        async def seed_lead_only() -> None:
            conn = await session_factory.connect()
            try:
                await conn.execute(
                    "INSERT INTO chat_session_profiles(session_id, display_name, org_name, interests_json) VALUES (?, ?, ?, ?)",
                    ("sess_lead_only", "王老师", "创新学校", json.dumps({}, ensure_ascii=False)),
                )
                await conn.execute(
                    "INSERT INTO lead_captures(session_id, contact_type, contact_value, visitor_type) VALUES (?, ?, ?, ?)",
                    ("sess_lead_only", "phone", "13900001111", "teacher"),
                )
                await conn.commit()
            finally:
                await conn.close()

        asyncio.run(seed_lead_only())

        list_resp = client.get("/admin-api/customers")
        assert list_resp.status_code == 200
        payload = list_resp.json()
        assert payload["total"] == 1
        assert payload["items"][0]["display_name"] == "王老师"
        assert payload["items"][0]["role_category"] == "老师"
        assert payload["items"][0]["contact"] == "13900001111"


def test_list_customers_deduplicates_chat_and_form_by_contact(tmp_path: Path) -> None:
    client = build_admin_customer_test_client(tmp_path)
    session_factory = DatabaseSessionFactory(str(tmp_path / "admin_customers.db"))

    with client:
        async def seed_chat_lead() -> None:
            conn = await session_factory.connect()
            try:
                await conn.execute(
                    "INSERT INTO chat_session_profiles(session_id, display_name, org_name, visitor_type, interests_json) VALUES (?, ?, ?, ?, ?)",
                    (
                        "sess_chat_same_phone",
                        "王老师",
                        "",
                        "teacher",
                        json.dumps(
                            {"_lead": {"contact_value": "18018278654", "contact_type": "phone"}},
                            ensure_ascii=False,
                        ),
                    ),
                )
                await conn.execute(
                    "INSERT INTO lead_captures(session_id, contact_type, contact_value, visitor_type) VALUES (?, ?, ?, ?)",
                    ("sess_chat_same_phone", "phone", "18018278654", "teacher"),
                )
                await conn.commit()
            finally:
                await conn.close()

        asyncio.run(seed_chat_lead())
        asyncio.run(
            _seed_trial_customer(
                session_factory,
                session_id="sess_form_same_phone",
                display_name="王校长",
                org_name="有为中学",
                contact="18018278654",
                email="zjy",
                visitor_type="institution_decision_maker",
            )
        )

        list_resp = client.get("/admin-api/customers")
        assert list_resp.status_code == 200
        payload = list_resp.json()
        assert payload["total"] == 1
        assert len(payload["items"]) == 1
        item = payload["items"][0]
        assert item["contact"] == "18018278654"
        assert item["org_name"] == "有为中学"
        assert item["email"] == "zjy"
        assert item["role_category"] in {"老师", "机构/学校负责人"}


def test_list_customers_includes_role_category_from_profile(tmp_path: Path) -> None:
    client = build_admin_customer_test_client(tmp_path)
    session_factory = DatabaseSessionFactory(str(tmp_path / "admin_customers.db"))

    with client:
        asyncio.run(
            _seed_trial_customer(
                session_factory,
                session_id="sess_role_1",
                display_name="刘校长",
                org_name="未来学校",
                visitor_type="institution_decision_maker",
            )
        )

        list_resp = client.get("/admin-api/customers")
        assert list_resp.status_code == 200
        payload = list_resp.json()
        assert payload["items"][0]["role_category"] == "机构/学校负责人"


def test_list_customers_includes_email_when_present(tmp_path: Path) -> None:
    client = build_admin_customer_test_client(tmp_path)
    session_factory = DatabaseSessionFactory(str(tmp_path / "admin_customers.db"))

    with client:
        asyncio.run(
            _seed_trial_customer(
                session_factory,
                session_id="sess_email_1",
                display_name="赵老师",
                org_name="实验小学",
                email="zhao@example.com",
            )
        )

        list_resp = client.get("/admin-api/customers")
        assert list_resp.status_code == 200
        item = list_resp.json()["items"][0]
        assert item["email"] == "zhao@example.com"


def test_list_customers_pagination(tmp_path: Path) -> None:
    client = build_admin_customer_test_client(tmp_path)
    session_factory = DatabaseSessionFactory(str(tmp_path / "admin_customers.db"))

    with client:
        for idx in range(12):
            asyncio.run(
                _seed_trial_customer(
                    session_factory,
                    session_id=f"sess_apply_{idx}",
                    display_name=f"客户{idx}",
                    org_name=f"单位{idx}",
                    contact=f"13800138{idx:03d}",
                )
            )

        page1 = client.get("/admin-api/customers", params={"page": 1, "page_size": 10})
        assert page1.status_code == 200
        payload1 = page1.json()
        assert payload1["total"] == 12
        assert payload1["total_pages"] == 2
        assert len(payload1["items"]) == 10

        page2 = client.get("/admin-api/customers", params={"page": 2, "page_size": 10})
        payload2 = page2.json()
        assert len(payload2["items"]) == 2


def test_new_customer_defaults_pending_statuses(tmp_path: Path) -> None:
    client = build_admin_customer_test_client(tmp_path)
    session_factory = DatabaseSessionFactory(str(tmp_path / "admin_customers.db"))

    with client:
        asyncio.run(_seed_trial_customer(session_factory, session_id="sess_defaults", display_name="赵老师", org_name="培训机构"))

        list_resp = client.get("/admin-api/customers")
        assert list_resp.status_code == 200
        item = list_resp.json()["items"][0]
        assert item["follow_up_status"] == "待跟进"
        assert item["trial_account"] == "待发放"


def test_update_test_account_status(tmp_path: Path) -> None:
    client = build_admin_customer_test_client(tmp_path)
    session_factory = DatabaseSessionFactory(str(tmp_path / "admin_customers.db"))

    with client:
        asyncio.run(_seed_trial_customer(session_factory, session_id="sess_test_acc", display_name="赵老师", org_name="培训机构"))

        patch_resp = client.patch(
            "/admin-api/customers/sess_test_acc/test-account",
            json={"test_account_status": "已发放"},
        )
        assert patch_resp.status_code == 200
        assert patch_resp.json()["trial_account"] == "已发放"

        summary_resp = client.get("/admin-api/customers/summary")
        assert summary_resp.json()["trial_issued_total"] == 1


def test_delete_customer_hides_from_list(tmp_path: Path) -> None:
    client = build_admin_customer_test_client(tmp_path)
    session_factory = DatabaseSessionFactory(str(tmp_path / "admin_customers.db"))

    with client:
        asyncio.run(_seed_trial_customer(session_factory, session_id="sess_delete_1", display_name="待删客户", org_name="测试单位"))

        async def seed_chat_history() -> None:
            conn = await session_factory.connect()
            try:
                await conn.execute(
                    "INSERT INTO chat_sessions(session_id, chat_mode, advisor_role) VALUES (?, ?, ?)",
                    ("sess_delete_1", "friend_v5", "friend"),
                )
                await conn.execute(
                    "INSERT INTO chat_messages(session_id, role, content) VALUES (?, ?, ?)",
                    ("sess_delete_1", "user", "我来咨询一下"),
                )
                await conn.commit()
            finally:
                await conn.close()

        asyncio.run(seed_chat_history())

        delete_resp = client.delete("/admin-api/customers/sess_delete_1")
        assert delete_resp.status_code == 200
        assert delete_resp.json()["ok"] is True

        list_resp = client.get("/admin-api/customers")
        assert list_resp.status_code == 200
        assert list_resp.json()["total"] == 0

        async def assert_hard_deleted() -> None:
            conn = await session_factory.connect()
            try:
                assert await conn.fetchval(
                    "SELECT COUNT(*) FROM lead_captures WHERE session_id=?",
                    ("sess_delete_1",),
                ) == 0
                assert await conn.fetchval(
                    "SELECT COUNT(*) FROM chat_session_profiles WHERE session_id=?",
                    ("sess_delete_1",),
                ) == 0
                assert await conn.fetchval(
                    "SELECT COUNT(*) FROM chat_messages WHERE session_id=?",
                    ("sess_delete_1",),
                ) == 0
                assert await conn.fetchval(
                    "SELECT COUNT(*) FROM chat_sessions WHERE session_id=?",
                    ("sess_delete_1",),
                ) == 0
            finally:
                await conn.close()

        asyncio.run(assert_hard_deleted())


def test_update_follow_up_status(tmp_path: Path) -> None:
    client = build_admin_customer_test_client(tmp_path)
    session_factory = DatabaseSessionFactory(str(tmp_path / "admin_customers.db"))

    with client:
        asyncio.run(_seed_trial_customer(session_factory, session_id="sess_apply_2", display_name="李主任", org_name="第一中学"))

        patch_resp = client.patch(
            "/admin-api/customers/sess_apply_2/follow-up",
            json={"follow_up_status": "已完成"},
        )
        assert patch_resp.status_code == 200
        assert patch_resp.json()["follow_up_status"] == "已完成"


def test_customer_summary_counts_trial_issued(tmp_path: Path) -> None:
    client = build_admin_customer_test_client(tmp_path)
    session_factory = DatabaseSessionFactory(str(tmp_path / "admin_customers.db"))

    with client:
        asyncio.run(
            _seed_trial_customer(
                session_factory,
                session_id="sess_trial_1",
                display_name="王校长",
                org_name="有为中学",
            )
        )

        summary_resp = client.get("/admin-api/customers/summary")
        assert summary_resp.status_code == 200
        payload = summary_resp.json()
        assert payload["customer_total"] == 1
        assert payload["trial_issued_total"] == 0

        client.patch(
            "/admin-api/customers/sess_trial_1/test-account",
            json={"test_account_status": "已发放"},
        )
        summary_resp2 = client.get("/admin-api/customers/summary")
        assert summary_resp2.json()["trial_issued_total"] == 1
