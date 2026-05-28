from __future__ import annotations

from typing import Optional

from app.conversation.skill_catalog import SKILL_CATALOG, SkillRoute  # noqa: F401 — re-export

# RAG 单选路由优先级（先匹配先返回）
_ROUTE_PRIORITY: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("stale-detector", ("过期", "陈旧", "检测", "stale", "更新建议", "健康度", "过期检测")),
    ("reading-digest", ("阅读笔记", "金句", "行动项", "核心观点", "digest")),
    ("daily-capture", ("碎片", "捕获", "记录", "待办", "想法收集", "daily", "capture")),
    ("note-refine", ("润色", "打磨", "refine", "优化表达", "改写", "提高质量", "note-refine")),
    ("knowledge-connect", ("关联", "联系", "聚类", "主题", "知识网络", "connect", "关联发现")),
    ("style-extract", ("风格", "用词", "句式", "表达习惯", "style", "画像", "style-extract")),
    ("smart-search", ("搜索", "找", "在哪里", "文档在哪", "smart-search", "查找")),
    (
        "smart-summary",
        ("总结", "摘要", "概述", "要点", "大概100字", "约100字", "一句话", "详细总结", "smart-summary"),
    ),
)


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(n and n in text for n in needles)


def route_skill(question: str) -> Optional[SkillRoute]:
    q = (question or "").strip()
    if not q:
        return None
    for skill_id, needles in _ROUTE_PRIORITY:
        if _contains_any(q, needles):
            entry = SKILL_CATALOG.get(skill_id)
            if entry:
                return entry
    return None
