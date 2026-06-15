from __future__ import annotations

from datetime import datetime
from pathlib import Path, PurePath
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from app.core.config import settings
from app.db.admin_customers import DEFAULT_PAGE_SIZE, AdminCustomerRepository, AdminCustomerRow
from app.db.repositories import AdminVideoAssetRepository, AdminVideoAssetRow
from app.schemas.admin import (
    AdminCustomerFollowUpUpdateRequest,
    AdminCustomerListResponse,
    AdminCustomerResponse,
    AdminCustomerSummaryResponse,
    AdminCustomerTestAccountUpdateRequest,
    AdminVideoAssetListResponse,
    AdminVideoAssetResponse,
)


router = APIRouter(prefix="/admin-api")

SCENES: dict[str, str] = {
    "general_ai_course": "人工智能通识课程",
    "project_based_learning": "跨学科项目式学习",
    "smart_enrollment": "智能招生",
    "school_ai_custom": "学校AI场景定制",
}

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm"}
ALLOWED_VIDEO_MIME_TYPES = {"video/mp4", "video/quicktime", "video/webm", "application/octet-stream"}


def get_admin_video_repository(request: Request) -> AdminVideoAssetRepository:
    repo = getattr(request.app.state, "admin_video_repository", None)
    if repo is None:
        raise HTTPException(status_code=503, detail="管理后台视频仓储未初始化")
    return repo


def get_admin_customer_repository(request: Request) -> AdminCustomerRepository:
    repo = getattr(request.app.state, "admin_customer_repository", None)
    if repo is None:
        raise HTTPException(status_code=503, detail="管理后台客户仓储未初始化")
    return repo


def get_admin_upload_dir(request: Request) -> Path:
    return Path(getattr(request.app.state, "admin_upload_dir", settings.admin_upload_dir))


@router.post("/videos/upload", response_model=AdminVideoAssetResponse)
async def upload_admin_video(
    request: Request,
    scene_key: str = Form(..., min_length=1, max_length=80),
    title: str = Form(default="", max_length=200),
    file: UploadFile = File(...),
    repo: AdminVideoAssetRepository = Depends(get_admin_video_repository),
) -> AdminVideoAssetResponse:
    scene = _normalize_scene(scene_key)
    original_filename = _safe_original_filename(file.filename or "video")
    extension = Path(original_filename).suffix.lower()
    mime_type = (file.content_type or "application/octet-stream").split(";")[0].strip().lower()
    if extension not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="仅支持 mp4、mov、webm 视频文件")
    if mime_type not in ALLOWED_VIDEO_MIME_TYPES:
        raise HTTPException(status_code=400, detail="视频文件类型不支持")

    upload_root = get_admin_upload_dir(request)
    scene_dir = upload_root / "videos" / scene_key
    scene_dir.mkdir(parents=True, exist_ok=True)

    stored_filename = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid4().hex[:12]}{extension}"
    target = (scene_dir / stored_filename).resolve()
    if scene_dir.resolve() not in target.parents:
        raise HTTPException(status_code=400, detail="非法视频保存路径")

    file_size = await _save_upload_file(file=file, target=target, max_bytes=settings.admin_video_max_bytes)
    rel_path = f"videos/{scene_key}/{stored_filename}"
    file_url = f"/admin-media/videos/{scene_key}/{stored_filename}"
    row = await repo.insert_video(
        scene_key=scene_key,
        scene_name=scene,
        title=(title or "").strip() or Path(original_filename).stem or original_filename,
        original_filename=original_filename,
        stored_filename=stored_filename,
        file_path=rel_path,
        file_url=file_url,
        mime_type=mime_type,
        file_size=file_size,
    )
    return _video_response(row)


@router.get("/videos", response_model=AdminVideoAssetListResponse)
async def list_admin_videos(
    scene_key: str | None = None,
    repo: AdminVideoAssetRepository = Depends(get_admin_video_repository),
) -> AdminVideoAssetListResponse:
    scene_filter = (scene_key or "").strip() or None
    if scene_filter:
        _normalize_scene(scene_filter)
    rows = await repo.list_videos(scene_key=scene_filter)
    return AdminVideoAssetListResponse(items=[_video_response(row) for row in rows])


@router.delete("/videos/{asset_id}")
async def delete_admin_video(
    asset_id: int,
    request: Request,
    repo: AdminVideoAssetRepository = Depends(get_admin_video_repository),
) -> dict[str, bool]:
    row = await repo.delete_video(asset_id=asset_id)
    if row is None:
        raise HTTPException(status_code=404, detail="视频不存在")
    upload_root = get_admin_upload_dir(request).resolve()
    target = (upload_root / row.file_path).resolve()
    if upload_root == target or upload_root not in target.parents:
        raise HTTPException(status_code=400, detail="非法视频保存路径")
    target.unlink(missing_ok=True)
    return {"ok": True}


@router.get("/customers/summary", response_model=AdminCustomerSummaryResponse)
async def customer_summary(
    repo: AdminCustomerRepository = Depends(get_admin_customer_repository),
) -> AdminCustomerSummaryResponse:
    customer_total, trial_issued_total = await _gather_customer_summary(repo)
    return AdminCustomerSummaryResponse(
        customer_total=customer_total,
        trial_issued_total=trial_issued_total,
    )


@router.get("/customers", response_model=AdminCustomerListResponse)
async def list_customers(
    q: str = "",
    page: int = 1,
    page_size: int = DEFAULT_PAGE_SIZE,
    repo: AdminCustomerRepository = Depends(get_admin_customer_repository),
) -> AdminCustomerListResponse:
    size = max(1, min(int(page_size), 50))
    page_num = max(1, int(page))
    items, total = await repo.list_customers(query=q, page=page_num, page_size=size)
    total_pages = (total + size - 1) // size if total > 0 else 0
    return AdminCustomerListResponse(
        items=[_customer_response(row) for row in items],
        total=total,
        page=page_num,
        page_size=size,
        total_pages=total_pages,
    )


@router.patch("/customers/{session_id}/follow-up", response_model=AdminCustomerResponse)
async def update_customer_follow_up(
    session_id: str,
    payload: AdminCustomerFollowUpUpdateRequest,
    repo: AdminCustomerRepository = Depends(get_admin_customer_repository),
) -> AdminCustomerResponse:
    row = await repo.update_follow_up(
        session_id=session_id,
        follow_up_status=payload.follow_up_status,
    )
    if row is None:
        raise HTTPException(status_code=404, detail="客户不存在")
    return _customer_response(row)


@router.patch("/customers/{session_id}/test-account", response_model=AdminCustomerResponse)
async def update_customer_test_account(
    session_id: str,
    payload: AdminCustomerTestAccountUpdateRequest,
    repo: AdminCustomerRepository = Depends(get_admin_customer_repository),
) -> AdminCustomerResponse:
    row = await repo.update_test_account(
        session_id=session_id,
        test_account_status=payload.test_account_status,
    )
    if row is None:
        raise HTTPException(status_code=404, detail="客户不存在")
    return _customer_response(row)


@router.delete("/customers/{session_id}")
async def delete_customer(
    session_id: str,
    repo: AdminCustomerRepository = Depends(get_admin_customer_repository),
) -> dict[str, bool]:
    deleted = await repo.delete_customer(session_id=session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="客户不存在")
    return {"ok": True}


async def _gather_customer_summary(repo: AdminCustomerRepository) -> tuple[int, int]:
    customer_total = await repo.count_customers()
    trial_issued_total = await repo.count_trial_issued()
    return customer_total, trial_issued_total


def _normalize_scene(scene_key: str) -> str:
    key = (scene_key or "").strip()
    scene = SCENES.get(key)
    if not scene:
        raise HTTPException(status_code=400, detail="未知课程场景")
    return scene


def _safe_original_filename(filename: str) -> str:
    name = PurePath(filename).name.strip()
    return name or "video"


async def _save_upload_file(*, file: UploadFile, target: Path, max_bytes: int) -> int:
    total = 0
    try:
        with target.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    out.close()
                    target.unlink(missing_ok=True)
                    raise HTTPException(status_code=413, detail="视频文件超过大小限制")
                out.write(chunk)
    finally:
        await file.close()
    if total <= 0:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="视频文件为空")
    return total


def _video_response(row: AdminVideoAssetRow) -> AdminVideoAssetResponse:
    return AdminVideoAssetResponse(
        id=row.id,
        scene_key=row.scene_key,
        scene_name=row.scene_name,
        title=row.title,
        original_filename=row.original_filename,
        stored_filename=row.stored_filename,
        file_url=row.file_url,
        mime_type=row.mime_type,
        file_size=row.file_size,
        duration_seconds=row.duration_seconds,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _customer_response(row: AdminCustomerRow) -> AdminCustomerResponse:
    return AdminCustomerResponse(
        session_id=row.session_id,
        display_name=row.display_name,
        org_name=row.org_name,
        contact=row.contact,
        follow_up_status=row.follow_up_status,
        trial_account=row.trial_account,
        updated_at=row.updated_at,
    )
