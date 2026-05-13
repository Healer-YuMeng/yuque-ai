from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.chat_api import router as chat_router
from app.core.config import settings
from app.core.logger import setup_logging
from app.data.yuque_loader import YuqueLoader
from app.db.repositories import ChatSessionRepository, DocumentRepository, LeadCaptureRepository, QALogRepository
from app.db.session import DatabaseSessionFactory
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

