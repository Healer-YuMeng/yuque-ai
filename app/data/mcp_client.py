from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.core.logger import get_logger

logger = get_logger(__name__)

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except Exception:  # pragma: no cover - optional dependency at runtime
    ClientSession = None  # type: ignore[assignment]
    StdioServerParameters = None  # type: ignore[assignment]
    stdio_client = None  # type: ignore[assignment]


class MCPClientError(RuntimeError):
    pass


@dataclass(frozen=True)
class MCPSearchResult:
    doc_id: str
    title: str
    url: str
    snippet: str


@dataclass(frozen=True)
class MCPDocMeta:
    doc_id: str
    title: str
    slug: str
    url: str


@dataclass(frozen=True)
class MCPTocNode:
    title: str
    level: int
    doc_id: str
    slug: str


class YuqueMCPClient:
    def __init__(
        self,
        *,
        command: str,
        args: str,
        repo_id: str,
        search_tool: str,
        get_doc_tool: str,
    ) -> None:
        self._command = command.strip()
        self._args = [part for part in args.split(" ") if part]
        self._repo_id = repo_id.strip()
        self._search_tool = search_tool
        self._get_doc_tool = get_doc_tool
        self._list_docs_tool = "yuque_list_docs"
        self._get_toc_tool = "yuque_get_toc"

    @property
    def enabled(self) -> bool:
        return bool(self._command)

    @property
    def search_tool(self) -> str:
        return self._search_tool

    @property
    def get_doc_tool(self) -> str:
        return self._get_doc_tool

    @property
    def list_docs_tool(self) -> str:
        return self._list_docs_tool

    @property
    def get_toc_tool(self) -> str:
        return self._get_toc_tool

    @property
    def repo_id(self) -> str:
        return self._repo_id

    @property
    def read_tools(self) -> List[str]:
        return [
            "yuque_get_user",
            "yuque_list_books",
            "yuque_get_book",
            self._list_docs_tool,
            self._get_doc_tool,
            self._get_toc_tool,
            self._search_tool,
            "yuque_list_notes",
            "yuque_get_note",
        ]

    async def search(self, query: str) -> List[MCPSearchResult]:
        if not self.enabled:
            return []
        # yuque-mcp 的 yuque_search 需要显式 type 参数：doc/repo
        result = await self._call_tool(self._search_tool, {"query": query, "type": "doc"})
        items = self._parse_search_results(result)
        if not self._repo_id:
            return items
        scoped: List[MCPSearchResult] = []
        for item in items:
            url = (item.url or "").strip("/")
            # 语雀文档 URL 常见形态：/group/repo/slug
            parts = url.split("/")
            if len(parts) >= 2:
                repo = f"{parts[0]}/{parts[1]}"
                if repo == self._repo_id:
                    scoped.append(item)
            elif not item.url:
                # 无 URL 时保守保留，避免误删可能可用结果
                scoped.append(item)
        return scoped

    async def get_doc(self, doc_id: str) -> str:
        if not self.enabled or not self._repo_id or not doc_id:
            return ""
        result = await self._call_tool(self._get_doc_tool, {"repo_id": self._repo_id, "doc_id": doc_id})
        if isinstance(result, dict):
            return str(result.get("body") or result.get("content") or "")
        return str(result)

    async def list_docs(self) -> List[MCPDocMeta]:
        if not self.enabled or not self._repo_id:
            return []
        result = await self._call_tool(self._list_docs_tool, {"repo_id": self._repo_id})
        return self._parse_docs_list(result)

    async def get_toc(self) -> List[MCPTocNode]:
        if not self.enabled or not self._repo_id:
            return []
        result = await self._call_tool(self._get_toc_tool, {"repo_id": self._repo_id})
        return self._parse_toc_nodes(result)

    async def get_user(self) -> Any:
        if not self.enabled:
            return {}
        return await self._call_tool("yuque_get_user", {})

    async def list_books(self) -> Any:
        if not self.enabled:
            return []
        return await self._call_tool("yuque_list_books", {})

    async def get_book(self, repo_id: str | None = None) -> Any:
        if not self.enabled:
            return {}
        return await self._call_tool("yuque_get_book", {"repo_id": repo_id or self._repo_id})

    async def list_notes(self) -> Any:
        if not self.enabled:
            return []
        return await self._call_tool("yuque_list_notes", {})

    async def get_note(self, note_id: str) -> Any:
        if not self.enabled:
            return {}
        return await self._call_tool("yuque_get_note", {"note_id": note_id})

    async def call_raw(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        if tool_name not in self.read_tools:
            raise MCPClientError(f"不允许调用工具: {tool_name}")
        return await self._call_tool(tool_name, arguments)

    async def _call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        if not self.enabled:
            return {}
        if not (ClientSession and StdioServerParameters and stdio_client):
            raise MCPClientError("未安装 mcp 依赖，无法调用 yuque-mcp-server。")

        server_params = StdioServerParameters(command=self._command, args=self._args)
        try:
            async with stdio_client(server_params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    response = await session.call_tool(tool_name, arguments)
        except Exception as exc:  # pragma: no cover - runtime integration
            logger.warning("mcp_call_failed tool=%s error=%s", tool_name, exc)
            raise MCPClientError(f"MCP 调用失败: {exc}") from exc

        content = getattr(response, "content", response)
        if isinstance(content, list) and content:
            text = getattr(content[0], "text", None)
            if text:
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return text
        return content

    @staticmethod
    def _parse_search_results(payload: Any) -> List[MCPSearchResult]:
        items: List[MCPSearchResult] = []
        if isinstance(payload, dict):
            raw_items = payload.get("items") or payload.get("results") or []
        elif isinstance(payload, list):
            raw_items = payload
        else:
            raw_items = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            target = item.get("target") if isinstance(item.get("target"), dict) else {}
            raw_doc_id = item.get("id") or item.get("doc_id") or target.get("id") or target.get("slug")
            items.append(
                MCPSearchResult(
                    doc_id=str(raw_doc_id or ""),
                    title=str(item.get("title") or ""),
                    url=str(item.get("url") or ""),
                    snippet=str(item.get("snippet") or item.get("summary") or ""),
                )
            )
        return items

    @staticmethod
    def _parse_docs_list(payload: Any) -> List[MCPDocMeta]:
        if isinstance(payload, dict):
            raw_items = payload.get("data") or payload.get("items") or payload.get("results") or []
        elif isinstance(payload, list):
            raw_items = payload
        else:
            raw_items = []
        out: List[MCPDocMeta] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            out.append(
                MCPDocMeta(
                    doc_id=str(item.get("id") or ""),
                    title=str(item.get("title") or ""),
                    slug=str(item.get("slug") or ""),
                    url=str(item.get("url") or ""),
                )
            )
        return out

    @staticmethod
    def _parse_toc_nodes(payload: Any) -> List[MCPTocNode]:
        if isinstance(payload, dict):
            raw_items = payload.get("data") or payload.get("items") or payload.get("results") or []
        elif isinstance(payload, list):
            raw_items = payload
        else:
            raw_items = []
        out: List[MCPTocNode] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            raw_level = item.get("depth") or item.get("level") or 1
            try:
                level = int(raw_level)
            except (TypeError, ValueError):
                level = 1
            out.append(
                MCPTocNode(
                    title=str(item.get("title") or ""),
                    level=level,
                    doc_id=str(item.get("doc_id") or item.get("id") or ""),
                    slug=str(item.get("slug") or ""),
                )
            )
        return out

