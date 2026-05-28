from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.api.chat_api import router as chat_router
from app.core.config import settings
from app.core.logger import setup_logging
from app.data.yuque_loader import YuqueLoader
from app.db.repositories import ChatSessionRepository, DocumentRepository, LeadCaptureRepository, QALogRepository
from app.db.session import DatabaseSessionFactory
from app.db.profile_repository import ChatSessionProfileRepository
from app.service.qa_service import QAService
from app.storage.vector_store import VectorStore


@asynccontextmanager
async def lifespan(application: FastAPI):
    setup_logging()
    session_factory = DatabaseSessionFactory(str(settings.sqlite_path))
    qa_service = QAService(
        yuque_loader=YuqueLoader(
            token=settings.yuque_token,
            base_url=settings.yuque_base_url,
            timeout_s=settings.yuque_timeout_s,
            scope=settings.yuque_scope,
        ),
        vector_store=VectorStore(vector_dir=settings.vector_dir),
        document_repository=DocumentRepository(session_factory),
        qa_log_repository=QALogRepository(session_factory),
        lead_capture_repository=LeadCaptureRepository(session_factory),
        chat_session_repository=ChatSessionRepository(session_factory),
        chat_session_profile_repository=ChatSessionProfileRepository(session_factory),
    )
    await qa_service.startup()
    application.state.qa_service = qa_service
    yield
    await qa_service.shutdown()


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(chat_router)

ui_dir = settings.frontend_dist_dir if settings.frontend_dist_dir.exists() else settings.web_dir
assets_dir = ui_dir / "assets"
if assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")


@app.get("/")
async def serve_index() -> FileResponse:
    return FileResponse(Path(ui_dir) / "index.html")


@app.get("/visitor")
async def serve_visitor() -> FileResponse:
    """
    访客专用入口：与根路径共用同一前端构建产物。

    前端会根据 window.location.pathname 判断是否为 /visitor，
    并按需隐藏开发者面板等仅面向内部使用的功能入口。
    """
    return FileResponse(Path(ui_dir) / "index.html")


@app.get("/youwei-logo.png")
async def serve_youwei_logo() -> FileResponse:
    """
    兼容开发/生产入口的 logo 访问。

    - Vite 开发态会直接托管 frontend/public。
    - FastAPI 托管 dist 时，public 文件可能在 dist 根目录；此处提供兜底。
    """
    dist_logo = settings.frontend_dist_dir / "youwei-logo.png"
    public_logo = settings.frontend_dir / "public" / "youwei-logo.png"
    if dist_logo.exists():
        return FileResponse(dist_logo)
    if public_logo.exists():
        return FileResponse(public_logo)
    raise HTTPException(status_code=404, detail="logo not found")


@app.get("/dev-links", response_class=HTMLResponse)
async def serve_dev_links() -> HTMLResponse:
    return HTMLResponse(
        """
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>开发快捷入口</title>
    <style>
      body {
        margin: 0;
        min-height: 100vh;
        display: flex;
        align-items: center;
        justify-content: center;
        background: #f5f7fb;
        color: #1f2937;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }
      .card {
        width: min(92vw, 560px);
        background: #fff;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 30px rgba(15, 23, 42, 0.08);
      }
      h1 {
        margin: 0 0 8px;
        font-size: 22px;
      }
      p {
        margin: 0 0 18px;
        color: #4b5563;
      }
      .links {
        display: grid;
        gap: 12px;
      }
      a {
        display: block;
        text-decoration: none;
        border: 1px solid #dbe2ea;
        border-radius: 12px;
        padding: 14px 16px;
        color: #111827;
        background: #f9fafb;
      }
      a:hover {
        border-color: #93c5fd;
        background: #eff6ff;
      }
      .title {
        font-weight: 700;
        margin-bottom: 4px;
      }
      .desc {
        font-size: 13px;
        color: #6b7280;
      }
    </style>
  </head>
  <body>
    <main class="card">
      <h1>开发快捷入口</h1>
      <p>建议固定开下面两个页面，对照开发与验收效果。</p>
      <section class="links">
        <a href="http://127.0.0.1:5173/" target="_blank" rel="noreferrer">
          <div class="title">开发者页面（5173）</div>
          <div class="desc">有开发者面板，前端热更新最快。</div>
        </a>
        <a href="http://127.0.0.1:8000/visitor" target="_blank" rel="noreferrer">
          <div class="title">访客页面（8000/visitor）</div>
          <div class="desc">用户视角页面，无开发者面板与上传图标。</div>
        </a>
      </section>
    </main>
  </body>
</html>
        """.strip()
    )

