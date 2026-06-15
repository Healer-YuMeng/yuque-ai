from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from app.api.admin_api import router as admin_router
from app.api.chat_api import router as chat_router
from app.api.chat_v5_api import router as chat_v5_router
from app.core.config import settings
from app.core.logger import setup_logging
from app.data.yuque_loader import YuqueLoader
from app.db.admin_customers import AdminCustomerRepository
from app.db.repositories import AdminVideoAssetRepository, ChatSessionRepository, DocumentRepository, LeadCaptureRepository, QALogRepository
from app.db.session import DatabaseSessionFactory
from app.db.profile_repository import ChatSessionProfileRepository
from app.service.qa_service import QAService
from app.storage.vector_store import VectorStore


@asynccontextmanager
async def lifespan(application: FastAPI):
    setup_logging()
    session_factory = DatabaseSessionFactory(str(settings.sqlite_path))
    admin_video_repository = AdminVideoAssetRepository(session_factory)
    admin_customer_repository = AdminCustomerRepository(session_factory)
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
        admin_video_asset_repository=admin_video_repository,
    )
    await qa_service.startup()
    await admin_video_repository.init_db()
    application.state.qa_service = qa_service
    application.state.admin_video_repository = admin_video_repository
    application.state.admin_customer_repository = admin_customer_repository
    application.state.admin_upload_dir = settings.admin_upload_dir
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
app.include_router(chat_v5_router)
app.include_router(admin_router)


def current_ui_dir() -> Path:
    return settings.frontend_dist_dir if settings.frontend_dist_dir.exists() else settings.web_dir


def serve_spa_index() -> FileResponse:
    return FileResponse(
        current_ui_dir() / "index.html",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/assets/{asset_path:path}")
async def serve_frontend_asset(asset_path: str) -> FileResponse:
    assets_dir = (current_ui_dir() / "assets").resolve()
    target = (assets_dir / asset_path).resolve()
    if assets_dir not in target.parents or not target.is_file():
        raise HTTPException(status_code=404, detail="asset not found")
    return FileResponse(
        target,
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@app.get("/admin-media/videos/{scene_key}/{filename}")
async def serve_admin_video(scene_key: str, filename: str) -> FileResponse:
    videos_dir = (settings.admin_upload_dir / "videos").resolve()
    target = (videos_dir / scene_key / filename).resolve()
    if videos_dir not in target.parents or not target.is_file():
        raise HTTPException(status_code=404, detail="video not found")
    return FileResponse(
        target,
        media_type=_video_media_type(target.suffix),
        headers={"Cache-Control": "public, max-age=86400"},
    )


def _video_media_type(suffix: str) -> str:
    ext = (suffix or "").lower()
    if ext == ".mp4":
        return "video/mp4"
    if ext == ".mov":
        return "video/quicktime"
    if ext == ".webm":
        return "video/webm"
    return "application/octet-stream"


@app.get("/")
async def serve_index() -> FileResponse:
    return serve_spa_index()


@app.get("/visitor")
async def serve_visitor() -> FileResponse:
    """
    访客专用入口：与根路径共用同一前端构建产物。

    前端会根据 window.location.pathname 判断是否为 /visitor，
    并按需隐藏开发者面板等仅面向内部使用的功能入口。
    """
    return serve_spa_index()


@app.get("/admin")
async def serve_admin() -> FileResponse:
    """
    管理后台入口：与其他前端路由共用同一前端构建产物。

    前端会根据 window.location.pathname 判断是否为 /admin，
    并切换到内部管理员使用的后台界面。
    """
    return serve_spa_index()


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
async def serve_dev_links(request: Request) -> HTMLResponse:
    host = request.headers.get("host") or request.url.netloc or "127.0.0.1"
    if ":" not in host and settings.frontend_public_port not in (80, 443):
        host = f"{host}:{settings.frontend_public_port}"
    display_port = host.rsplit(":", 1)[1] if ":" in host else ("443" if request.url.scheme == "https" else "80")
    frontend_origin = f"{request.url.scheme}://{host}"
    html = """
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
        <a href="__FRONTEND_ORIGIN__/" target="_blank" rel="noreferrer">
          <div class="title">开发者页面（__FRONTEND_PORT__）</div>
          <div class="desc">有开发者面板，用当前部署端口访问。</div>
        </a>
        <a href="__FRONTEND_ORIGIN__/visitor" target="_blank" rel="noreferrer">
          <div class="title">访客页面（__FRONTEND_PORT__/visitor）</div>
          <div class="desc">用户视角页面，无开发者面板与上传图标。</div>
        </a>
        <a href="__FRONTEND_ORIGIN__/admin" target="_blank" rel="noreferrer">
          <div class="title">管理后台（__FRONTEND_PORT__/admin）</div>
          <div class="desc">管理员使用的工作台与知识库管理页面。</div>
        </a>
      </section>
    </main>
  </body>
</html>
        """.strip()
    return HTMLResponse(
        html.replace("__FRONTEND_ORIGIN__", frontend_origin).replace(
            "__FRONTEND_PORT__",
            display_port,
        )
    )
