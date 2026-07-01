from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from app.schemas.chat import ChatMediaBundle


FriendV5Scene = Literal[
    "人工智能通识教育",
    "智能招生",
    "跨学科项目化学习",
    "学校AI场景定制",
]
FriendV5TriggerType = Literal["scene", "tag", "manual"]


class ChatV5Request(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    chat_mode: Literal["friend_v5"] = "friend_v5"
    session_id: str = Field(..., min_length=1, max_length=120)
    scene: FriendV5Scene
    trigger_type: FriendV5TriggerType
    model: Optional[str] = Field(default=None, min_length=1, max_length=120)
    owner: Optional[str] = Field(default=None, min_length=1, max_length=120)
    token_profile: Optional[str] = Field(default=None, max_length=24)

    @field_validator("session_id")
    @classmethod
    def _require_v5_session_prefix(cls, value: str) -> str:
        sid = (value or "").strip()
        if not sid.startswith("sess_v5_"):
            raise ValueError("V5 session_id 必须以 sess_v5_ 开头。")
        return sid

    @field_validator("model", "owner", "token_profile", mode="before")
    @classmethod
    def _empty_str_to_none(cls, value: Any) -> Any:
        if value == "":
            return None
        return value


class FriendV5SourceItem(BaseModel):
    source_type: Literal["web", "yuque"]
    title: str = ""
    url: Optional[str] = None
    snippet: Optional[str] = None
    index: Optional[int] = None
    doc_id: Optional[str] = None


class ChatV5DonePayload(BaseModel):
    answer: str
    tags: List[str] = Field(default_factory=list)
    sources: List[FriendV5SourceItem] = Field(default_factory=list)
    search_keywords: List[str] = Field(default_factory=list)
    media: ChatMediaBundle = Field(default_factory=ChatMediaBundle)
    profile_fields: Dict[str, Any] = Field(default_factory=dict)
    trial_apply_available: bool = False
    fallback_used: bool = False
    debug: Optional[Dict[str, Any]] = None


class ChatV5CapabilitiesResponse(BaseModel):
    enabled: bool
    model: str
    require_web_sources: bool = True
