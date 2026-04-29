from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class DocSuggestRequest(BaseModel):
    # 支持前端仅输入 '@' 的场景：query 可能为空
    query: str = Field(..., min_length=0, max_length=120)
    owner: Optional[str] = Field(default=None, min_length=1, max_length=120)


class DocMeta(BaseModel):
    id: Optional[int] = None
    slug: Optional[str] = None
    title: str
    url: Optional[str] = None
    updated_at: Optional[str] = None


class DocSuggestResponse(BaseModel):
    docs: List[DocMeta]

