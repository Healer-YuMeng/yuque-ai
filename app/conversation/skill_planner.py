from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from app.conversation.skill_catalog import SALES_SKILL_IDS, SKILL_CATALOG
from app.conversation.skill_catalog import SkillRoute


@dataclass(frozen=True)
class PlannedSkill:
    skill_id: str
    reason: str
    generation_instruction: str

    @classmethod
    def from_route(cls, route: SkillRoute, *, reason: str) -> PlannedSkill:
        return cls(
            skill_id=route.skill_id,
            reason=reason,
            generation_instruction=route.generation_instruction,
        )


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(n and n in text for n in needles)


def plan_sales_skills(
    question: str,
    *,
    catalog_path: Sequence[str] | None = None,
    dialog_level: int = 0,
    max_skills: int = 3,
) -> List[PlannedSkill]:
    """销售 V4：动态选择 0~N 个 Skill（仅拼 prompt，不单独跑流水线）。"""
    q = (question or "").strip()
    if not q or dialog_level <= 1:
        return []

    picked: List[PlannedSkill] = []
    seen: set[str] = set()

    def _add(skill_id: str, reason: str) -> None:
        if skill_id in seen or skill_id not in SALES_SKILL_IDS:
            return
        route = SKILL_CATALOG.get(skill_id)
        if not route:
            return
        seen.add(skill_id)
        picked.append(PlannedSkill.from_route(route, reason=reason))

    if _contains_any(q, ("总结", "摘要", "概述", "要点", "大概100字", "约100字", "一句话", "简要", "介绍", "是什么")):
        _add("smart-summary", "总结/概述类问题")

    if _contains_any(q, ("搜索", "找", "在哪里", "文档在哪", "查找", "有哪些", "有没有")):
        _add("smart-search", "查找/列举类问题")

    if _contains_any(q, ("关联", "联系", "对比", "区别", "关系", "和", "与", "相比", "有什么不同")):
        _add("knowledge-connect", "关联/对比类问题")

    if _contains_any(q, ("阅读笔记", "金句", "行动项", "核心观点", "笔记")):
        _add("reading-digest", "用户需要笔记式整理")

    # 深度讲解默认给摘要能力（若无其它 skill）
    if not picked and dialog_level >= 2:
        _add("smart-summary", "目录内深度讲解默认摘要结构")

    if len(picked) >= 2 and "smart-summary" in seen and "smart-search" in seen:
        # 搜索+摘要同时命中时保留两者；若过多则按优先级截断
        pass

    return picked[: max(1, int(max_skills))]


def format_skill_instructions_block(skills: Sequence[PlannedSkill]) -> str:
    if not skills:
        return ""
    lines = ["【本轮 Skill 约束（合并执行，勿分多次回答）】"]
    for s in skills:
        lines.append(f"- [{s.skill_id}] {s.reason}")
        lines.append(s.generation_instruction.strip())
    return "\n".join(lines) + "\n"
