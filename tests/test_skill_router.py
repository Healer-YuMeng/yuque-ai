from __future__ import annotations

from app.rag.skill_router import route_skill


def test_route_skill_smart_summary() -> None:
    r = route_skill("帮我总结一下这篇内容，约100字")
    assert r is not None
    assert r.skill_id == "smart-summary"
    assert "100" in r.generation_instruction


def test_route_skill_stale_detector() -> None:
    r = route_skill("请做一下过期检测，给我更新建议")
    assert r is not None
    assert r.skill_id == "stale-detector"


def test_route_skill_smart_search() -> None:
    r = route_skill("帮我搜索一下这个文档在哪里")
    assert r is not None
    assert r.skill_id == "smart-search"

