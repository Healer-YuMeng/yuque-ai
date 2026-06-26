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


class MediaItem(BaseModel):
    url: str
    title: str = ""
    doc_title: str = ""
    doc_id: Optional[str] = None
    summary: str = ""


class ChatMediaBundle(BaseModel):
    images: List[MediaItem] = Field(default_factory=list)
    videos: List[MediaItem] = Field(default_factory=list)


class ChatV2Request(ChatRequest):
    """V1.5 多媒体优先链路请求（继承旧参数，保持调用成本最低）。"""


class ChatV2Response(ChatResponse):
    answer_style: Literal["short_sales"] = "short_sales"
    media: ChatMediaBundle = Field(default_factory=ChatMediaBundle)
    lead_nudge_triggered: bool = False


class ChatV3Request(ChatRequest):
    """V3 访客销售引导链路（会话画像 + 兴趣推荐 + follow-up 优先答复）。"""


class ChatV3Response(ChatV2Response):
    """V3 返回结构沿用 V2，便于前端复用 media/lead 字段。"""


class ChatV3CapabilitiesResponse(BaseModel):
    enabled: bool
    toc_loaded: bool = False
    profile_enabled: bool = True


class ChatV4Request(ChatRequest):
    """V4：语雀目录状态机 + 目录内关联讲解（旁路）。"""


class ChatV4Response(ChatV2Response):
    """V4 返回结构沿用 V2（media / sources），并扩展试用入口。"""

    trial_apply_available: bool = False


class TrialCredentialsRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=120)


class TrialCredentialsResponse(BaseModel):
    ok: bool
    username: str = ""
    password: str = ""
    label: str = ""
    message: str = ""


class VisitorTrialApplyRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=120)
    name: str = Field(..., min_length=1, max_length=80)
    org_name: str = Field(..., min_length=1, max_length=160)
    contact: str = Field(default="", max_length=120)
    email: str = Field(default="", max_length=160)
    interested_product: str = Field(default="", max_length=160)
    concern: str = Field(default="", max_length=500)


class VisitorProfileResponse(BaseModel):
    ok: bool = True
    name: str = ""
    org_name: str = ""
    contact: str = ""
    email: str = ""
    interested_product: str = ""
    concern: str = ""
    module_scope: str = ""
    trial_account_issued: bool = False


class ChatV4CapabilitiesResponse(BaseModel):
    enabled: bool
    toc_loaded: bool = False
    catalog_state_enabled: bool = True


class GuideDocTitleNode(BaseModel):
    uuid: str = ""
    title: str
    level: int = 1
    parent_uuid: str = ""
    node_type: str = ""
    url: Optional[str] = None
    doc_id: Optional[int] = None
    children: List["GuideDocTitleNode"] = Field(default_factory=list)


class GuideDocTitlesResponse(BaseModel):
    v15_enabled: bool
    count: int
    titles: List[str] = Field(default_factory=list)
    total_nodes: int = 0
    root_nodes: int = 0
    max_level: int = 0
    nodes: List[GuideDocTitleNode] = Field(default_factory=list)
    refresh_interval_s: int
    refreshed_seconds_ago: Optional[float] = None


class ResetSessionRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=120)
    chat_mode: Literal["visitor_sales", "rag", "friend_v5"] = "visitor_sales"


class RebuildIndexResponse(BaseModel):
    indexed_documents: int
    indexed_chunks: int


class HealthResponse(BaseModel):
    status: str


class RuntimeModeResponse(BaseModel):
    mode: Literal["rag", "direct_yuque"]
    label: str
    llm_model: str = ""


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
