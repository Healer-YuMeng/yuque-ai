from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.admin_api import router
from app.db.repositories import AdminSceneIntroRepository, AdminVideoAssetRepository
from app.db.session import DatabaseSessionFactory


def build_admin_scene_intro_client(tmp_path: Path) -> TestClient:
    app = FastAPI()
    db_path = tmp_path / "admin_scene_intro.db"
    upload_dir = tmp_path / "uploads"
    session_factory = DatabaseSessionFactory(str(db_path))
    video_repo = AdminVideoAssetRepository(session_factory)
    intro_repo = AdminSceneIntroRepository(session_factory)

    @app.on_event("startup")
    async def _startup() -> None:
        await video_repo.init_db()
        await intro_repo.init_db()
        app.state.admin_video_repository = video_repo
        app.state.admin_scene_intro_repository = intro_repo
        app.state.admin_upload_dir = upload_dir

    app.include_router(router)
    client = TestClient(app)
    client.post("/admin-api/auth/login", json={"username": "admin", "password": "admin123456"})
    return client


def test_list_scene_intros_returns_all_scene_slots(tmp_path: Path) -> None:
    client = build_admin_scene_intro_client(tmp_path)

    with client:
        response = client.get("/admin-api/scene-intros")

    assert response.status_code == 200
    payload = response.json()
    keys = [item["scene_key"] for item in payload["items"]]
    assert keys == [
        "general_ai_course",
        "project_based_learning",
        "smart_enrollment",
        "school_ai_custom",
    ]


def test_update_scene_intro_persists_content(tmp_path: Path) -> None:
    client = build_admin_scene_intro_client(tmp_path)

    with client:
        save = client.put(
            "/admin-api/scene-intros/project_based_learning",
            json={
                "intro_text": "这是后台维护的 IDEAS-PBL 产品介绍。",
                "decision_intro_text": "这是给校长看的决策者介绍。",
                "user_intro_text": "这是给老师和家长看的使用者介绍。",
            },
        )
        assert save.status_code == 200
        assert save.json()["intro_text"] == "这是后台维护的 IDEAS-PBL 产品介绍。"
        assert save.json()["decision_intro_text"] == "这是给校长看的决策者介绍。"
        assert save.json()["user_intro_text"] == "这是给老师和家长看的使用者介绍。"

        listed = client.get("/admin-api/scene-intros")
        assert listed.status_code == 200
        target = next(item for item in listed.json()["items"] if item["scene_key"] == "project_based_learning")
        assert target["intro_text"] == "这是后台维护的 IDEAS-PBL 产品介绍。"
        assert target["decision_intro_text"] == "这是给校长看的决策者介绍。"
        assert target["user_intro_text"] == "这是给老师和家长看的使用者介绍。"
