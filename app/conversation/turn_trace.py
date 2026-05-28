from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

from app.core.config import settings


class McpCallTrace(BaseModel):
    tool: str
    query: str = ""
    doc_id: str = ""
    title: str = ""
    hit_count: int = 0
    body_chars: int = 0


class SkillTraceItem(BaseModel):
    skill_id: str
    reason: str = ""


class DocumentTraceItem(BaseModel):
    doc_id: str = ""
    title: str = ""
    role: str = "related"  # primary | related
    source_type: str = "mcp"
    snippet: str = ""


class TurnTrace(BaseModel):
    pipeline: str = "v4_content"
    catalog_path: List[str] = Field(default_factory=list)
    dialog_level: int = 0
    mcp_calls: List[McpCallTrace] = Field(default_factory=list)
    skills: List[SkillTraceItem] = Field(default_factory=list)
    documents: List[DocumentTraceItem] = Field(default_factory=list)


class TurnTraceBuilder:
    """V4 单轮运行追踪：MCP / Skill / 文档。"""

    def __init__(
        self,
        *,
        pipeline: str,
        catalog_path: Sequence[str] | None = None,
        dialog_level: int = 0,
    ) -> None:
        self._pipeline = pipeline
        self._catalog_path = list(catalog_path or [])
        self._dialog_level = int(dialog_level)
        self._mcp_calls: List[McpCallTrace] = []
        self._skills: List[SkillTraceItem] = []
        self._documents: List[DocumentTraceItem] = []

    def record_search(self, *, query: str, hit_count: int) -> None:
        self._mcp_calls.append(
            McpCallTrace(tool="yuque_search", query=(query or "").strip(), hit_count=max(0, int(hit_count)))
        )

    def record_get_doc(self, *, doc_id: str, title: str, body_chars: int) -> None:
        self._mcp_calls.append(
            McpCallTrace(
                tool="yuque_get_doc",
                doc_id=(doc_id or "").strip(),
                title=(title or "").strip(),
                body_chars=max(0, int(body_chars)),
            )
        )

    def set_skills(self, items: Sequence[SkillTraceItem]) -> None:
        self._skills = list(items)

    def add_document(
        self,
        *,
        doc_id: str,
        title: str,
        role: str,
        snippet: str = "",
        source_type: str = "mcp",
    ) -> None:
        self._documents.append(
            DocumentTraceItem(
                doc_id=(doc_id or "").strip(),
                title=(title or "").strip(),
                role=(role or "related").strip() or "related",
                source_type=(source_type or "mcp").strip() or "mcp",
                snippet=(snippet or "").strip()[:200],
            )
        )

    def build(self) -> TurnTrace:
        return TurnTrace(
            pipeline=self._pipeline,
            catalog_path=self._catalog_path,
            dialog_level=self._dialog_level,
            mcp_calls=list(self._mcp_calls),
            skills=list(self._skills),
            documents=list(self._documents),
        )

    def attach_debug(self, dbg: Dict[str, Any]) -> Dict[str, Any]:
        if not settings.expose_turn_trace:
            return dbg
        out = dict(dbg)
        out["turn_trace"] = self.build().model_dump()
        return out


def empty_guide_trace(
    *,
    pipeline: str,
    catalog_path: Sequence[str] | None = None,
    dialog_level: int = 0,
) -> Dict[str, Any]:
    builder = TurnTraceBuilder(pipeline=pipeline, catalog_path=catalog_path, dialog_level=dialog_level)
    return builder.attach_debug({})
