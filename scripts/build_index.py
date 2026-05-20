from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv

from app.core.config import settings
from app.data.yuque_loader import YuqueLoader
from app.db.repositories import DocumentRepository, LeadCaptureRepository, QALogRepository
from app.db.session import DatabaseSessionFactory
from app.service.qa_service import QAService
from app.storage.vector_store import VectorStore


async def main() -> None:
    load_dotenv()
    bootstrap_query = (os.getenv("BOOTSTRAP_QUERY") or "退款").strip()
    session_factory = DatabaseSessionFactory(str(settings.sqlite_path))
    service = QAService(
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
    )
    await service.startup()
    docs, chunks = await service.rebuild_index(bootstrap_query=bootstrap_query)
    print(f"indexed_documents={docs} indexed_chunks={chunks}")
    await service.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

