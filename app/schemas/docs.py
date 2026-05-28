from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class DocSuggestRequest(BaseModel):
    # 支持前端仅输入 '@' 的场景：query 可能为空
    query: str = Field(..., min_length=0, max_length=120)
    owner: Optional[str] = Field(default=None, min_length=1, max_length=120)
    token_profile: Optional[str] = Field(default=None, max_length=24)


class DocMeta(BaseModel):
    id: Optional[int] = None
    slug: Optional[str] = None
    title: str
    url: Optional[str] = None
    updated_at: Optional[str] = None
    # 语雀目录 toc：层级与节点类型（@ 联想展示目录树）
    toc_uuid: Optional[str] = None
    toc_parent_uuid: Optional[str] = None
    toc_level: Optional[int] = None
    toc_kind: Optional[str] = None  # "doc" | "title"
    toc_selectable: Optional[bool] = None


class DocSuggestResponse(BaseModel):
    docs: List[DocMeta]

