from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.admin_api import router
from app.db.repositories import AdminVideoAssetRepository
from app.db.session import DatabaseSessionFactory


def build_admin_test_client(tmp_path: Path) -> TestClient:
    app = FastAPI()
    db_path = tmp_path / "admin.db"
    upload_dir = tmp_path / "uploads"
    repo = AdminVideoAssetRepository(DatabaseSessionFactory(str(db_path)))

    @app.on_event("startup")
    async def _startup() -> None:
        await repo.init_db()
        app.state.admin_video_repository = repo
        app.state.admin_upload_dir = upload_dir

    app.include_router(router)
    client = TestClient(app)
    client.post("/admin-api/auth/login", json={"username": "admin", "password": "admin123456"})
    return client


def test_admin_video_upload_requires_login(tmp_path: Path) -> None:
    app = FastAPI()
    db_path = tmp_path / "admin.db"
    upload_dir = tmp_path / "uploads"
    repo = AdminVideoAssetRepository(DatabaseSessionFactory(str(db_path)))

    @app.on_event("startup")
    async def _startup() -> None:
        await repo.init_db()
        app.state.admin_video_repository = repo
        app.state.admin_upload_dir = upload_dir

    app.include_router(router)
    client = TestClient(app)

    with client:
        response = client.post(
            "/admin-api/videos/upload",
            data={"scene_key": "school_ai_custom", "title": "学校演示"},
            files={"file": ("demo.mp4", b"demo", "video/mp4")},
        )

    assert response.status_code == 401


def test_admin_login_sets_session_cookie(tmp_path: Path) -> None:
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    bad = client.post("/admin-api/auth/login", json={"username": "admin", "password": "wrong"})
    assert bad.status_code == 401

    ok = client.post("/admin-api/auth/login", json={"username": "admin", "password": "admin123456"})
    assert ok.status_code == 200
    assert ok.json()["authenticated"] is True

    status = client.get("/admin-api/auth/status")
    assert status.status_code == 200
    assert status.json()["authenticated"] is True


def test_upload_video_saves_file_and_returns_public_url(tmp_path: Path) -> None:
    client = build_admin_test_client(tmp_path)
    content = b"\x00\x00\x00\x18ftypmp42demo"

    with client:
        response = client.post(
            "/admin-api/videos/upload",
            data={"scene_key": "school_ai_custom", "title": "学校演示"},
            files={"file": ("demo.mp4", content, "video/mp4")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["scene_key"] == "school_ai_custom"
    assert payload["scene_name"] == "学校AI场景定制"
    assert payload["title"] == "学校演示"
    assert payload["file_url"].startswith("/admin-media/videos/school_ai_custom/")
    saved = tmp_path / "uploads" / "videos" / "school_ai_custom" / payload["stored_filename"]
    assert saved.read_bytes() == content


def test_list_videos_filters_by_scene(tmp_path: Path) -> None:
    client = build_admin_test_client(tmp_path)

    with client:
        client.post(
            "/admin-api/videos/upload",
            data={"scene_key": "school_ai_custom"},
            files={"file": ("school.mp4", b"school", "video/mp4")},
        )
        client.post(
            "/admin-api/videos/upload",
            data={"scene_key": "smart_enrollment"},
            files={"file": ("enroll.webm", b"enroll", "video/webm")},
        )
        response = client.get("/admin-api/videos", params={"scene_key": "school_ai_custom"})

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["items"]) == 1
    assert payload["items"][0]["scene_key"] == "school_ai_custom"
    assert payload["items"][0]["original_filename"] == "school.mp4"


def test_delete_video_marks_record_deleted_and_removes_file(tmp_path: Path) -> None:
    client = build_admin_test_client(tmp_path)

    with client:
        upload = client.post(
            "/admin-api/videos/upload",
            data={"scene_key": "school_ai_custom", "title": "待删除视频"},
            files={"file": ("school.mp4", b"school", "video/mp4")},
        )
        assert upload.status_code == 200
        payload = upload.json()
        saved = tmp_path / "uploads" / "videos" / "school_ai_custom" / payload["stored_filename"]
        assert saved.exists()

        response = client.delete(f"/admin-api/videos/{payload['id']}")
        assert response.status_code == 200
        assert response.json() == {"ok": True}
        assert not saved.exists()

        listed = client.get("/admin-api/videos", params={"scene_key": "school_ai_custom"})
        assert listed.status_code == 200
        assert listed.json()["items"] == []

        second = client.delete(f"/admin-api/videos/{payload['id']}")
        assert second.status_code == 404
