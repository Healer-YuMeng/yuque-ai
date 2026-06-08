from __future__ import annotations

import json

import pytest

from app.rag.friend_v5_generator import FriendV5Generator


class _FakeDashScopeStream:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    async def __aenter__(self) -> "_FakeDashScopeStream":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        return None

    def raise_for_status(self) -> None:
        return None

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakeDashScopeClient:
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines
        self.payload: dict | None = None
        self.headers: dict | None = None

    def stream(self, method: str, url: str, *, headers: dict, json: dict, timeout: float):  # noqa: A002
        self.method = method
        self.url = url
        self.headers = headers
        self.payload = json
        self.timeout = timeout
        return _FakeDashScopeStream(self.lines)


def _sse_payload(payload: dict) -> str:
    return "data: " + json.dumps(payload, ensure_ascii=False)


@pytest.mark.asyncio
async def test_friend_v5_generator_sends_enable_search() -> None:
    fake = _FakeDashScopeClient(
        [
            _sse_payload(
                {"choices": [{"delta": {"content": "你好"}, "finish_reason": None, "index": 0}]}
            ),
            "data: [DONE]",
        ]
    )
    generator = FriendV5Generator(api_key="sk-test", client=fake, require_web_sources=False)

    events = [item async for item in generator.stream(system_prompt="系统", user_prompt="问题")]

    assert fake.payload["model"] == "qwen3.7-plus"
    assert fake.payload["enable_thinking"] is False
    assert fake.payload["enable_search"] is False
    assert fake.payload["search_options"]["enable_source"] is True
    assert fake.payload["search_options"]["search_strategy"] == "turbo"
    assert "X-DashScope-SSE" not in fake.headers
    assert any(item.event == "token" and item.token == "你好" for item in events)


@pytest.mark.asyncio
async def test_friend_v5_generator_no_web_sources_from_oai_text() -> None:
    fake = _FakeDashScopeClient(
        [
            _sse_payload(
                {"choices": [{"delta": {"content": "文本 [来源：https://fake.com] 继续"}, "finish_reason": None, "index": 0}]}
            ),
            "data: [DONE]",
        ]
    )
    generator = FriendV5Generator(api_key="sk-test", client=fake, require_web_sources=True)
    events = [item async for item in generator.stream(system_prompt="系统", user_prompt="问题")]

    assert any(item.event == "token" for item in events)
    assert not any(item.event == "web_sources" for item in events)
