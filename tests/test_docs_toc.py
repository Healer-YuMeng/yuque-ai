from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import chat_api
from app.api.chat_api import router
from app.schemas.docs import DocMeta


@pytest.fixture()
def toc_client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


async def _fake_fetch_toc_metas(_owner: str | None, token_profile: str | None = None) -> list[DocMeta]:
    return [
        DocMeta(
            id=101,
            slug="hello",
            title="Hello Doc",
            toc_uuid="u1",
            toc_level=1,
            toc_kind="doc",
            toc_selectable=True,
        ),
        DocMeta(
            id=None,
            slug=None,
            title="分组 A",
            toc_uuid="u2",
            toc_level=1,
            toc_kind="title",
            toc_selectable=False,
        ),
    ]


def test_docs_toc_returns_metas(toc_client: TestClient) -> None:
    with patch.object(chat_api, "_fetch_toc_doc_metas", side_effect=_fake_fetch_toc_metas):
        r = toc_client.get("/docs/toc", params={"owner": "demo-login"})
    assert r.status_code == 200
    data = r.json()
    assert len(data["docs"]) == 2
    assert data["docs"][0]["id"] == 101
    assert data["docs"][0]["slug"] == "hello"
    assert data["docs"][1]["toc_selectable"] is False


def test_docs_toc_503_on_loader_error(toc_client: TestClient) -> None:
    from app.data.yuque_loader import YuqueLoaderError

    async def boom(_owner: str | None, token_profile: str | None = None) -> list[DocMeta]:
        raise YuqueLoaderError("token bad")

    with patch.object(chat_api, "_fetch_toc_doc_metas", side_effect=boom):
        r = toc_client.get("/docs/toc")
    assert r.status_code == 503
    assert "token bad" in r.json()["detail"]
