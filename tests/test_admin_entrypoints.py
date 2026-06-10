from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi.responses import FileResponse

from app import main as main_module


def test_admin_route_returns_spa_index(monkeypatch) -> None:
    sentinel = FileResponse(Path(__file__))
    monkeypatch.setattr(main_module, "serve_spa_index", lambda: sentinel)

    response = asyncio.run(main_module.serve_admin())

    assert response is sentinel


def test_frontend_main_switches_admin_to_dedicated_app() -> None:
    content = (Path(__file__).resolve().parents[1] / "frontend" / "src" / "main.tsx").read_text(encoding="utf-8")

    assert "import AdminApp from './AdminApp.tsx'" in content
    assert 'window.location.pathname.startsWith("/admin")' in content
    assert "rootComponent === AdminApp ? <AdminApp />" in content
