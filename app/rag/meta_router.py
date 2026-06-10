from __future__ import annotations

import json
import re
from typing import Any, Tuple

from openai import AsyncOpenAI

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

# 明显是在问业务/文档的长问句，跳过元问题 LLM（省一次调用）
_DOC_QUERY_HINTS = (
    "文档",
    "语雀",
    "知识库",
    "第",
    "条",
    "流程",
    "政策",
    "规定",
    "办理",
    "申请",
    "报销",
    "退款",
    "发票",
    "登录",
    "密码",
    "如何",
    "怎么",
    "是否支持",
    "研究生",
    "平台",
)


def _likely_knowledge_question(question: str) -> bool:
    """启发式：更像在查资料则不做元问题 LLM，减少误伤与费用。"""
    t = (question or "").strip()
    if len(t) > settings.assistant_meta_router_max_chars:
        return True
    return any(h in t for h in _DOC_QUERY_HINTS)


async def assistant_only_by_llm(question: str) -> Tuple[bool, str]:
    """
    单次 LLM 调用：判断用户是否**仅在**问助手自身（能力/边界/功能），而非索要语雀里的业务事实。
    返回 (assistant_only, reason)，reason 为 llm|skipped_long|skipped_doc_hint|no_key|parse_error|error
    """
    q = (question or "").strip()
    if not q:
        return False, "empty"
    if len(q) > settings.assistant_meta_router_max_chars:
        return False, "skipped_long"
    if _likely_knowledge_question(q):
        return False, "skipped_doc_hint"
    model = (settings.assistant_meta_router_model or settings.intent_llm_model or settings.llm_model).strip()
    key, base = settings.resolve_model_endpoint(model)
    key = (key or "").strip()
    if not key:
        return False, "no_key"
    base = (base or "").strip()
    if not base:
        return False, "no_base_url"
    client = AsyncOpenAI(api_key=key, base_url=base or None)
    system = (
        "你是路由分类器。判断用户这句话是不是**只在问聊天助手/本系统本身**（能做什么、有哪些功能、"
        "能否回答库外问题、使用范围、和谁对比、你是谁等），**而不是**在查询公司知识库里的具体业务、政策、流程、某篇文档写了什么。\n"
        "若用户混合了「助手能做什么」和「某业务怎么办」，判为 knowledge。\n"
        "只输出一行 JSON，不要 markdown，不要解释："
        '{"assistant_only":true} 或 {"assistant_only":false}'
    )
    raw = ""
    try:
        resp = await client.chat.completions.create(
            model=model,
            temperature=0,
            extra_body={"enable_thinking": False},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": q[:500]},
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE)
        data: Any = json.loads(raw)
        if isinstance(data, dict) and "assistant_only" in data:
            v = data["assistant_only"]
            if isinstance(v, bool):
                return v, "llm"
            if str(v).lower() in ("true", "1", "yes"):
                return True, "llm"
            return False, "llm"
        return False, "parse_error"
    except json.JSONDecodeError as exc:
        snippet = raw[:120] if raw else ""
        logger.info("assistant_meta_router_parse_error raw=%r err=%s", snippet, exc)
        return False, "parse_error"
    except Exception as exc:
        logger.warning("assistant_meta_router_failed err=%s", exc)
        return False, "error"


async def should_use_direct_assistant_help(retrieval_question: str) -> Tuple[bool, str]:
    """
    是否应跳过检索、直接使用内置助手说明。
    返回 (yes, reason)：reason 含 rule（正则命中）/ llm / none
    """
    from app.rag.scope_help import is_assistant_scope_meta_question

    if is_assistant_scope_meta_question(retrieval_question):
        return True, "rule"
    if not settings.assistant_meta_llm_router:
        return False, "router_disabled"
    only, why = await assistant_only_by_llm(retrieval_question)
    if only:
        return True, f"llm:{why}"
    return False, f"llm:{why}"
