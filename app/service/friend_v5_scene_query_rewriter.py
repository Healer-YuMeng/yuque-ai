from __future__ import annotations

import json
from typing import Any, Optional, Sequence

import httpx

from app.core.logger import get_logger
from app.rag.generator import GeneratorConfigError

logger = get_logger(__name__)


class FriendV5SceneQueryRewriter:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        generation_url: str,
        timeout_s: float = 30.0,
        client: Optional[Any] = None,
    ) -> None:
        self._api_key = (api_key or "").strip()
        self._model = (model or "").strip()
        self._generation_url = (generation_url or "").strip()
        self._timeout_s = float(timeout_s)
        self._client = client

    async def rewrite(self, *, question: str, scene: str, toc_nodes: Sequence[dict[str, Any]]) -> str:
        if not self._api_key:
            raise GeneratorConfigError("缺少 DASHSCOPE_API_KEY，无法执行 V5 场景查询改写。")
        if not self._generation_url:
            raise GeneratorConfigError("缺少 CHAT_V5_GENERATION_URL，无法执行 V5 场景查询改写。")

        prompt = self._build_user_prompt(question=question, scene=scene, toc_nodes=toc_nodes)
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是语雀知识库检索改写助手。"
                        "请把用户当前问题改写成更适合语雀文档标题/目录匹配的检索词。"
                        "不要解释，不要回答问题，只输出 JSON。"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "max_tokens": 120,
            "enable_thinking": False,
            "enable_search": False,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        own_client = self._client is None
        client = self._client or httpx.AsyncClient()
        try:
            response = await client.post(
                self._generation_url,
                headers=headers,
                json=payload,
                timeout=self._timeout_s,
            )
            response.raise_for_status()
            data = response.json()
        finally:
            if own_client and hasattr(client, "aclose"):
                await client.aclose()

        rewritten = self._extract_rewritten_query(data)
        if rewritten:
            return rewritten
        logger.info("scene_query_rewrite_empty fallback_to_question scene=%s question=%s", scene, question[:120])
        return (question or scene).strip()

    @staticmethod
    def _build_user_prompt(*, question: str, scene: str, toc_nodes: Sequence[dict[str, Any]]) -> str:
        toc_titles = [str(item.get("title") or "").strip() for item in toc_nodes if str(item.get("title") or "").strip()]
        toc_block = "\n".join(f"- {title}" for title in toc_titles[:20]) or "（无）"
        return (
            "请把下面这轮场景咨询，改写成一条适合语雀文档检索的查询词。\n"
            "要求：\n"
            "1. 保留场景主语义；\n"
            "2. 尽量贴近语雀目录/文档标题风格；\n"
            "3. 可补充 2-4 个高相关关键词；\n"
            "4. 不要编造成句答案；\n"
            '5. 仅输出 JSON，格式为 {"query":"..."}。\n\n'
            f"场景：{scene}\n"
            f"用户原话：{question}\n"
            f"可参考的目录标题：\n{toc_block}\n"
        )

    @staticmethod
    def _extract_rewritten_query(payload: dict[str, Any]) -> str:
        choices = payload.get("choices") or payload.get("output", {}).get("choices") or []
        text = ""
        if isinstance(choices, list) and choices:
            choice = choices[0] if isinstance(choices[0], dict) else {}
            message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
            text = str(message.get("content") or choice.get("content") or "").strip()
        if not text:
            output = payload.get("output") if isinstance(payload.get("output"), dict) else {}
            text = str(output.get("text") or "").strip()
        if not text:
            return ""
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return text
        if not isinstance(data, dict):
            return ""
        return str(data.get("query") or "").strip()
