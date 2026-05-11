from __future__ import annotations

import re
from dataclasses import replace
from typing import List

from app.rag.retriever import RetrievalResult
from app.schemas.chat import SourceItem

# 注入给 LLM 的说明：非语雀正文；用于「不问具体文档」的元问题，**不调用**语雀 MCP/检索。
ASSISTANT_META_HELP_CONTEXT = """【助手能力与范围（系统说明，非从语雀某篇文档检索；回答用户时请用 Markdown 分条说明，勿声称下列未列出的能力）】

## 一、定位与边界
1）本助手以**语雀知识库**为主要依据：检索到相关文档后，结合大模型组织语言作答，并尽量列出参考来源。
2）知识库未收录或检索无命中时，**不编造**具体业务、政策或人事类事实。
3）关于「知识库以外 / 其他泛问」：以知识库内可引用内容为主；未收录议题请补充语雀文档或通过组织正式渠道确认。

## 二、可提供的能力概览（实现以当前部署配置为准）
- **知识问答**：就组织内文档（流程、制度、FAQ 等）提问，支持流式回答。
- **检索方式**：向量检索（若已配置 embedding 与索引）、语雀 API 直连检索；可选 **yuque-mcp-server** 做补充检索（由环境变量控制，非每问必调）。
- **知识库目录 / 文档列表**：在启用 MCP 时，可拉取目录树（TOC）与文档列表，便于了解知识库结构；也可在问句中要求层级与统计表（与合并清单逻辑配合）。
- **技能（Skill）提示**：根据问题关键词注入只读「技能」说明（如阅读摘要、过期检测等），仍基于检索到的上下文生成，**不会写回语雀**。
- **不适用**：多租户权限管理、代用户直接改语雀正文、保证库外实时新闻准确性等，均不在本 MVP 范围内。

## 三、如何提问更有效
尽量包含业务关键词、文档可能标题中的词；需要列表或目录时，可明确要求「文档列表」「目录结构」等。"""

# 兼容旧名称（测试与外部引用）
SCOPE_HELP_CONTEXT = ASSISTANT_META_HELP_CONTEXT


def contexts_effectively_empty(contexts: List[str]) -> bool:
    return not contexts or all(not (c or "").strip() for c in contexts)


def is_assistant_scope_meta_question(question: str) -> bool:
    """识别「不问具体语雀文档」的元问题：能力边界、能否问库外、能回答哪些类问题等；命中后 pipeline 将不调 MCP/检索。"""
    t = (question or "").strip()
    if not t or len(t) > 240:
        return False

    # 「你可以回答哪些问题」——要求「回答/解答」后紧接「哪些/什么+问题」，避免「回答退款…需要准备哪些问题」误命中
    if re.search(
        r"(你可以|你能|能否|可不可以|会不会|是否|能不能|可否|是否可以)"
        r".{0,12}(回答|解答)\s*[：:，,]?\s*(哪些|什么)问题",
        t,
    ):
        return True
    if re.search(
        r"(你|本助手|这个助手).{0,8}(能|会).{0,12}(做什么|干什么|干嘛|有哪些能力|有什么功能|支持什么|会什么)",
        t,
    ):
        return True
    if re.search(r"(你|本助手).{0,6}(有|能|会).{0,8}什么功能", t):
        return True
    # 「你有哪些功能」——与「什么功能」不同，避免仅靠宽松正则误伤长句
    compact = re.sub(r"\s+", "", t)
    if any(
        phrase in compact
        for phrase in (
            "你有哪些功能",
            "你有啥功能",
            "你都有啥功能",
            "本助手有哪些功能",
            "这个助手有哪些功能",
            "你会哪些功能",
        )
    ):
        return True
    if "功能列表" in t and len(t) <= 80:
        return True

    # 泛问：「你可以回答其他问题吗」——不含「知识库」字样，需单独匹配，且要求「其他问题/别的问题」整块，避免误伤「其他部门」
    if re.search(
        r"(你可以|你能|能否|可不可以|会不会|是否|能不能|可否|是否可以)"
        r".{0,20}(回答|解答).{0,10}(其他问题|别的问题|另外的问题|其他内容|别的内容)",
        t,
    ):
        return True

    outer = any(
        p in t
        for p in (
            "知识库以外",
            "知识库之外",
            "语雀以外",
            "语雀之外",
            "库外",
            "不在知识库",
            "没进知识库",
            "知识库里没有",
        )
    )
    if not outer:
        return False
    # 避免「知识库以外的某业务政策」类事实问句：通常不含「你可以/能否…回答」
    if re.search(
        r"(你可以|你能|能否|可不可以|会不会|是否|能不能|可否|是否可以)"
        r".{0,22}(回答|解答|问|查询)",
        t,
    ):
        return True
    if re.search(r"(回答|解答).{0,18}(知识库|语雀).{0,10}(以外|之外)", t):
        return True
    return "你可以回答知识库以外" in compact


def direct_scope_help_retrieval(*, route: str = "rule") -> RetrievalResult:
    """不调语雀/MCP/向量：直接返回能力与范围说明（含功能列表要点），避免元问题误走检索。
    route: rule=关键词/短语命中；llm=由 ASSISTANT_META_LLM_ROUTER 分类命中。
    """
    snippet = ASSISTANT_META_HELP_CONTEXT[:200]
    src = SourceItem(
        title="助手能力与范围（系统说明）",
        url=None,
        source_type="yuque",
        snippet=snippet,
    )
    return RetrievalResult(
        contexts=[ASSISTANT_META_HELP_CONTEXT],
        sources=[src],
        fallback_used=False,
        debug={
            "retrieval_mode": "scope_help_direct",
            "scope_help_bypass_retrieval": True,
            "scope_help_injected": True,
            "assistant_meta_route": route,
        },
    )


def apply_scope_help_if_needed(retrieval: RetrievalResult, user_question: str) -> RetrievalResult:
    """已走正常检索后，若仍无正文上下文且为元问题，则注入说明（兜底；优先由 pipeline 短路）。"""
    if not contexts_effectively_empty(retrieval.contexts):
        return retrieval
    if not is_assistant_scope_meta_question(user_question):
        return retrieval
    snippet = ASSISTANT_META_HELP_CONTEXT[:200]
    src = SourceItem(
        title="助手能力与范围（系统说明）",
        url=None,
        source_type="yuque",
        snippet=snippet,
    )
    new_debug = dict(retrieval.debug)
    new_debug["scope_help_injected"] = True
    return replace(
        retrieval,
        contexts=[ASSISTANT_META_HELP_CONTEXT],
        sources=[src],
        debug=new_debug,
    )
