from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class SourceItem(BaseModel):
    title: str
    url: Optional[str] = None
    source_type: Literal["vector", "mcp", "yuque"]
    snippet: Optional[str] = None
    score: Optional[float] = None


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    # 前端可选：模型与语雀所有者（用于动态路由/作用域）
    model: Optional[str] = Field(default=None, min_length=1, max_length=120)
    owner: Optional[str] = Field(default=None, min_length=1, max_length=120)


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    fallback_used: bool = False
    debug: Optional[Dict[str, Any]] = None


class RebuildIndexResponse(BaseModel):
    indexed_documents: int
    indexed_chunks: int


class HealthResponse(BaseModel):
    status: str


class RuntimeModeResponse(BaseModel):
    mode: Literal["rag", "direct_yuque"]
    label: str


class MCPToolItem(BaseModel):
    name: str
    category: str
    status: Literal["integrated", "available"]
    description: str


class MCPCapabilitiesResponse(BaseModel):
    enabled: bool
    repo_scope: str
    tools: List[MCPToolItem]

