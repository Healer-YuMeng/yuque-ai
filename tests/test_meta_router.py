from __future__ import annotations

from dataclasses import replace

import pytest

import app.core.config as cfg
from app.rag.meta_router import (
    _likely_knowledge_question,
    assistant_only_by_llm,
    should_use_direct_assistant_help,
)


def test_likely_knowledge_question_hints() -> None:
    assert _likely_knowledge_question("如何办理请假")
    assert _likely_knowledge_question("研发平台登录")
    assert not _likely_knowledge_question("你是谁")


@pytest.mark.asyncio
async def test_should_use_direct_rule_hit() -> None:
    ok, reason = await should_use_direct_assistant_help("你有哪些功能？")
    assert ok is True
    assert reason == "rule"


@pytest.mark.asyncio
async def test_should_use_direct_router_disabled() -> None:
    ok, reason = await should_use_direct_assistant_help("你是谁呀")
    assert ok is False
    assert reason == "router_disabled"


@pytest.mark.asyncio
async def test_should_use_direct_llm_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    new_settings = replace(cfg.settings, assistant_meta_llm_router=True)
    monkeypatch.setattr("app.rag.meta_router.settings", new_settings)

    async def _fake_llm(_q: str) -> tuple[bool, str]:
        return True, "llm"

    monkeypatch.setattr("app.rag.meta_router.assistant_only_by_llm", _fake_llm)
    ok, reason = await should_use_direct_assistant_help("没写进正则的元问题短句")
    assert ok is True
    assert reason == "llm:llm"


@pytest.mark.asyncio
async def test_assistant_only_skipped_doc_hint() -> None:
    only, why = await assistant_only_by_llm("如何申请报销")
    assert only is False
    assert why == "skipped_doc_hint"
