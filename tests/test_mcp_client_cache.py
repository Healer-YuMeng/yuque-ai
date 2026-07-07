from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

import app.data.mcp_client as mcp_module
from app.data.mcp_client import YuqueMCPClient


class _FakeText:
    def __init__(self, payload: Any) -> None:
        self.text = json.dumps(payload)


class _FakeResponse:
    def __init__(self, payload: Any) -> None:
        self.content = [_FakeText(payload)]


class _FakeClientSession:
    calls: list[tuple[str, dict[str, Any]]] = []

    def __init__(self, read_stream: Any, write_stream: Any) -> None:
        self.read_stream = read_stream
        self.write_stream = write_stream

    async def __aenter__(self) -> "_FakeClientSession":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    async def initialize(self) -> None:
        return None

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> _FakeResponse:
        self.calls.append((tool_name, arguments))
        return _FakeResponse(
            {
                "items": [
                    {
                        "id": "doc-1",
                        "title": "人工智能通识课程",
                        "url": "https://www.yuque.com/login/repo/doc-1",
                        "snippet": "课程简介",
                    }
                ]
            }
        )


class _FakeStdioContext:
    async def __aenter__(self) -> tuple[object, object]:
        return object(), object()

    async def __aexit__(self, *args: Any) -> None:
        return None


def _fake_stdio_client(params: Any) -> _FakeStdioContext:
    return _FakeStdioContext()


@pytest.mark.asyncio
async def test_mcp_search_reuses_cached_response_for_same_request(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeClientSession.calls = []
    monkeypatch.setattr(mcp_module, "ClientSession", _FakeClientSession)
    monkeypatch.setattr(mcp_module, "StdioServerParameters", lambda command, args: (command, args))
    monkeypatch.setattr(mcp_module, "stdio_client", _fake_stdio_client)
    monkeypatch.setattr(mcp_module, "settings", SimpleNamespace(mcp_cache_ttl_s=60.0, mcp_timeout_s=20.0))
    YuqueMCPClient.clear_cache()

    client = YuqueMCPClient(
        command="npx",
        args="-y yuque-mcp",
        repo_id="login/repo",
        search_tool="yuque_search",
        get_doc_tool="yuque_get_doc",
    )

    first = await client.search("人工智能")
    second = await client.search("人工智能")

    assert first == second
    assert len(_FakeClientSession.calls) == 1
