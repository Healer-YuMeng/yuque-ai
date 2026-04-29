from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


SkillId = str


@dataclass(frozen=True)
class SkillRoute:
    skill_id: SkillId
    # 该指令会注入到“生成问题”里，让模型按照 skill 输出格式回答。
    generation_instruction: str


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(n and n in text for n in needles)


def route_skill(question: str) -> Optional[SkillRoute]:
    q = (question or "").strip()
    if not q:
        return None

    # stale-detector（必须更靠前，避免“总结/概述”误触发）
    if _contains_any(q, ("过期", "陈旧", "检测", "stale", "更新建议", "健康度", "过期检测")):
        return SkillRoute(
            skill_id="stale-detector",
            generation_instruction=(
                "你正在执行 yuque-personal/stale-detector（只读）。\n"
                "你的任务：基于上下文中提供的“文档标题+更新时间”，判断哪些文档可能过期/需要更新。\n"
                "输出要求：\n"
                "1) 以列表给出疑似过期文档（按风险从高到低，最多 10 条）。\n"
                "2) 每条包含：标题、更新时间、为何可能过期（2-3 点）、建议怎么更新（1-2 点）。\n"
                "3) 如果上下文不足以判断，则明确说明“无法判断”，并给出建议的最少信息（例如需要查看正文）。"
            ),
        )

    # reading-digest
    if _contains_any(q, ("阅读笔记", "金句", "行动项", "核心观点", "digest")):
        return SkillRoute(
            skill_id="reading-digest",
            generation_instruction=(
                "你正在执行 yuque-personal/reading-digest（只读）。\n"
                "你的任务：基于上下文总结阅读内容，输出结构化阅读笔记。\n"
                "输出要求：\n"
                "1) 核心观点（3 条）：每条一句话 + 关键依据。\n"
                "2) 金句（3 条）：从上下文提炼的精炼句子。\n"
                "3) 行动项（3 条）：可执行的下一步建议。\n"
                "4) 若上下文不足，请只输出“无法确定”的部分，并说明缺了什么。"
            ),
        )

    # daily-capture
    if _contains_any(q, ("碎片", "捕获", "记录", "待办", "想法收集", "daily", "capture")):
        return SkillRoute(
            skill_id="daily-capture",
            generation_instruction=(
                "你正在执行 yuque-personal/daily-capture（只读）。\n"
                "你的任务：把用户输入的碎片想法/待办整理成结构化条目，便于保存到语雀。\n"
                "输出要求：\n"
                "1) 建议标题（1 条）。\n"
                "2) 精炼正文（3-6 句）：包含关键信息与背景。\n"
                "3) 标签（3-8 个），用 # 开头。\n"
                "4) 待办（1-3 个），每个一句话。\n"
                "仅输出结构化结果，不要写“已保存到语雀”（因为这是只读模式）。"
            ),
        )

    # note-refine
    if _contains_any(q, ("润色", "打磨", "refine", "优化表达", "改写", "提高质量", "note-refine")):
        return SkillRoute(
            skill_id="note-refine",
            generation_instruction=(
                "你正在执行 yuque-personal/note-refine（只读）。\n"
                "你的任务：在不改变原意的前提下润色并提升结构。\n"
                "输出要求：\n"
                "1) 输出“优化后的全文”。\n"
                "2) 如果上下文含有原文，请保留关键事实与引用点。\n"
                "3) 不要承诺写回语雀，只输出结果内容。"
            ),
        )

    # knowledge-connect
    if _contains_any(q, ("关联", "联系", "聚类", "主题", "知识网络", "connect", "关联发现")):
        return SkillRoute(
            skill_id="knowledge-connect",
            generation_instruction=(
                "你正在执行 yuque-personal/knowledge-connect（只读）。\n"
                "你的任务：基于上下文的多文档内容，找出主题关联并组织成知识网络视角。\n"
                "输出要求：\n"
                "1) 按主题簇分组（最多 5 簇）。\n"
                "2) 每簇包含：主题名称 + 1-2 句共性总结。\n"
                "3) 给出每簇的关键关联点（2-4 条），并用上下文/引用支持。\n"
                "4) 如果只能推断而缺乏证据，明确标注“推测”。"
            ),
        )

    # style-extract
    if _contains_any(q, ("风格", "用词", "句式", "表达习惯", "style", "画像", "style-extract")):
        return SkillRoute(
            skill_id="style-extract",
            generation_instruction=(
                "你正在执行 yuque-personal/style-extract（只读）。\n"
                "你的任务：从上下文样本文档中提炼写作风格画像。\n"
                "输出要求：\n"
                "1) 语气：偏正式/口语/学术/叙述？给 1 句话解释。\n"
                "2) 句式：短句/长句比例、常见连接词/过渡方式（给例子）。\n"
                "3) 结构：常见小标题/段落组织。\n"
                "4) 术语与标点偏好。\n"
                "若上下文不足，说明证据不足并给出需要补充的材料类型。"
            ),
        )

    # smart-search
    if _contains_any(q, ("搜索", "找", "在哪里", "文档在哪", "smart-search", "查找")):
        return SkillRoute(
            skill_id="smart-search",
            generation_instruction=(
                "你正在执行 yuque-personal/smart-search（只读）。\n"
                "你的任务：把检索到的候选上下文组织成“可读的搜索回答”。\n"
                "输出要求：\n"
                "1) 列出候选文档标题（最多 5 个）。\n"
                "2) 给出每个候选的 1-2 句摘要，说明和用户问题的相关点。\n"
                "3) 只根据上下文，不要编造。\n"
                "4) 最终在参考来源里保留引用（由系统基于 sources 自动生成）。"
            ),
        )

    # smart-summary
    if _contains_any(q, ("总结", "摘要", "概述", "要点", "大概100字", "约100字", "一句话", "详细总结", "smart-summary")):
        return SkillRoute(
            skill_id="smart-summary",
            generation_instruction=(
                "你正在执行 yuque-personal/smart-summary（只读）。\n"
                "你的任务：按用户要求的粒度生成摘要。\n"
                "输出要求：\n"
                "1) 如果用户提到“一句话/简要/概述”，先给一句话。\n"
                "2) 再给要点列表（3-6 条）。\n"
                "3) 如用户要求“详细/要点+详细”，再给 1 段详细说明。\n"
                "4) 如果用户明确要约 100 字，尽量控制在该范围。\n"
                "5) 不要编造上下文没有的信息。"
            ),
        )

    return None

