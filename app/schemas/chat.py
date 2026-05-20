from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class SelectedYuqueDocRef(BaseModel):
    """前端从目录或 @ 联想选择的语雀文档，用于检索锚定。"""

    doc_id: int = Field(..., ge=1)
    slug: Optional[str] = Field(default=None, max_length=200)
    title: Optional[str] = Field(default=None, max_length=500)


class SourceItem(BaseModel):
    title: str
    url: Optional[str] = None
    source_type: Literal["vector", "mcp", "yuque"]
    snippet: Optional[str] = None
    score: Optional[float] = None
    doc_id: Optional[str] = None


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    # visitor_sales：有为 AI 销售咨询（默认）；rag：原知识库问答形态（含参考来源结构）
    chat_mode: Literal["visitor_sales", "rag"] = "visitor_sales"
    # 访客会话标识，用于 lead_captures 去重与关联；建议前端每会话 UUID
    session_id: Optional[str] = Field(default=None, min_length=1, max_length=120)
    # 前端可选：模型与语雀所有者（用于动态路由/作用域）
    model: Optional[str] = Field(default=None, min_length=1, max_length=120)
    owner: Optional[str] = Field(default=None, min_length=1, max_length=120)
    # 语雀 Token 档位：primary（默认）或 secondary（.env 中 YUQUE_TOKEN_SECONDARY）
    token_profile: Optional[str] = Field(default=None, max_length=24)
    # 从目录 / @ 选择的多篇文档；优先按 doc_id 拉取正文锚定检索
    selected_yuque_docs: Optional[List[SelectedYuqueDocRef]] = Field(default=None, max_length=20)

    @field_validator("session_id", "model", "owner", "token_profile", mode="before")
    @classmethod
    def _empty_str_to_none(cls, value: Any) -> Any:
        """前端常发 \"\"；Optional+min_length=1 会把空串判为非法，统一视为未传。"""
        if value == "":
            return None
        return value


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    fallback_used: bool = False
    debug: Optional[Dict[str, Any]] = None


class ChatMessageItem(BaseModel):
    role: Literal["user", "assistant"]
    text: str
    created_at: str


class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: List[ChatMessageItem]


class ResetSessionRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=120)
    chat_mode: Literal["visitor_sales", "rag"] = "visitor_sales"


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
    """含主/副 Token 各自 /user 的 login，供前端区分账号（可与 YUQUE_SCOPE 中的 login 不同）。"""

    enabled: bool
    repo_scope: str
    secondary_token_configured: bool = False
    repo_scope_secondary: str = ""
    yuque_token_primary_login: str = ""
    yuque_token_secondary_login: str = ""
    # True：.env 已单独填写非空 YUQUE_SCOPE_SECONDARY；False：repo_scope_secondary 由主库回退
    yuque_scope_secondary_explicit: bool = False
    tools: List[MCPToolItem]

