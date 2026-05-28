from __future__ import annotations

from app.conversation.skill_planner import plan_sales_skills


def test_plan_sales_skills_multi_intent() -> None:
    skills = plan_sales_skills(
        "跨学科项目式学习是什么，和通识课有什么关系",
        catalog_path=["案例与社区", "跨学科项目式学习"],
        dialog_level=2,
    )
    ids = {s.skill_id for s in skills}
    assert "smart-summary" in ids
    assert "knowledge-connect" in ids
    assert len(skills) <= 3


def test_plan_sales_skills_guide_level_empty() -> None:
    assert plan_sales_skills("平台介绍", dialog_level=1) == []


def test_plan_sales_skills_excludes_non_sales() -> None:
    skills = plan_sales_skills("帮我检测过期文档", dialog_level=2)
    assert all(s.skill_id in {"smart-summary", "smart-search", "knowledge-connect", "reading-digest"} for s in skills)


def test_plan_sales_skills_reading_digest_only_when_asked() -> None:
    skills = plan_sales_skills("请给我阅读笔记和金句", dialog_level=2)
    assert any(s.skill_id == "reading-digest" for s in skills)
