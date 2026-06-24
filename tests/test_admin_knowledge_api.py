from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import admin_api
from app.data.yuque_loader import YuqueDocument, YuqueTocNode


class _FakeYuqueLoader:
    async def get_book_toc(self, *, book: str):
        assert book == "demo/book"
        return [
            YuqueTocNode(
                uuid="root",
                type="TITLE",
                title="平台介绍",
                url="",
                doc_id=None,
                level=1,
                parent_uuid="",
            ),
            YuqueTocNode(
                uuid="doc-1",
                type="DOC",
                title="人工智能通识课程",
                url="https://www.yuque.com/demo/book/ai-course",
                doc_id=101,
                level=2,
                parent_uuid="root",
            ),
        ]

    async def get_doc(self, *, book: str, id_or_slug: str):
        assert book == "demo/book"
        assert id_or_slug == "101"
        return YuqueDocument(
            doc_id="101",
            title="人工智能通识课程",
            url="https://www.yuque.com/demo/book/ai-course",
            body="这里是语雀正文。",
        )

    async def close(self) -> None:
        return None


def _build_client(monkeypatch) -> TestClient:
    app = FastAPI()
    app.include_router(admin_api.router)
    monkeypatch.setattr(admin_api, "_build_admin_yuque_loader", lambda **_kwargs: (_FakeYuqueLoader(), "demo/book"))
    client = TestClient(app)
    client.post("/admin-api/auth/login", json={"username": "admin", "password": "admin123456"})
    return client


def test_admin_knowledge_toc_returns_yuque_nodes(monkeypatch) -> None:
    client = _build_client(monkeypatch)

    resp = client.get("/admin-api/knowledge/toc")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["scope"] == "demo/book"
    assert payload["items"][0]["selectable"] is False
    assert payload["items"][1]["title"] == "人工智能通识课程"
    assert payload["items"][1]["doc_id"] == "101"
    assert payload["items"][1]["selectable"] is True


def test_admin_knowledge_doc_returns_body(monkeypatch) -> None:
    client = _build_client(monkeypatch)

    resp = client.get("/admin-api/knowledge/docs/101")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["title"] == "人工智能通识课程"
    assert payload["body"] == "这里是语雀正文。"
