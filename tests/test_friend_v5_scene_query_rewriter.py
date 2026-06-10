from __future__ import annotations

import json

import httpx
import pytest

from app.service.friend_v5_scene_query_rewriter import FriendV5SceneQueryRewriter


@pytest.mark.asyncio
async def test_scene_query_rewriter_returns_query_from_json_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({"query": "智能招生 招生问答示例 AI获客"})
                    }
                }
            ]
        }
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    rewriter = FriendV5SceneQueryRewriter(
        api_key="test-key",
        model="qwen3.7-plus",
        generation_url="https://example.com/chat/completions",
        client=client,
    )

    result = await rewriter.rewrite(
        question="我想了解招生怎么做",
        scene="智能招生",
        toc_nodes=[{"title": "招生问答示例"}, {"title": "课程介绍"}],
    )

    assert result == "智能招生 招生问答示例 AI获客"
    await client.aclose()


@pytest.mark.asyncio
async def test_scene_query_rewriter_falls_back_to_original_question_when_query_missing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = {"choices": [{"message": {"content": json.dumps({"note": "missing"})}}]}
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    rewriter = FriendV5SceneQueryRewriter(
        api_key="test-key",
        model="qwen3.7-plus",
        generation_url="https://example.com/chat/completions",
        client=client,
    )

    result = await rewriter.rewrite(
        question="我想了解招生怎么做",
        scene="智能招生",
        toc_nodes=[],
    )

    assert result == "我想了解招生怎么做"
    await client.aclose()
