from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class AdminVideoAssetResponse(BaseModel):
    id: int
    scene_key: str
    scene_name: str
    title: str
    original_filename: str
    stored_filename: str
    file_url: str
    mime_type: str
    file_size: int
    duration_seconds: int | None = None
    created_at: str = ""
    updated_at: str = ""


class AdminVideoAssetListResponse(BaseModel):
    items: List[AdminVideoAssetResponse]


class AdminCustomerResponse(BaseModel):
    session_id: str
    display_name: str = ""
    org_name: str = ""
    contact: str = ""
    follow_up_status: str = "待跟进"
    trial_account: str = "待发放"
    updated_at: str = ""


class AdminCustomerListResponse(BaseModel):
    items: List[AdminCustomerResponse]
    total: int = 0
    page: int = 1
    page_size: int = 10
    total_pages: int = 0


class AdminCustomerSummaryResponse(BaseModel):
    customer_total: int = 0
    trial_issued_total: int = 0


class AdminCustomerFollowUpUpdateRequest(BaseModel):
    follow_up_status: str = Field(..., min_length=1, max_length=40)


class AdminCustomerTestAccountUpdateRequest(BaseModel):
    test_account_status: str = Field(..., min_length=1, max_length=40)
