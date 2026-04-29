from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator, Sequence

from openai import AsyncOpenAI

from app.schemas.chat import SourceItem


class GeneratorConfigError(RuntimeError):
    pass


class Generator(ABC):
    @abstractmethod
    async def generate(self, *, question: str, contexts: Sequence[str], sources: Sequence[SourceItem]) -> str:
        raise NotImplementedError

    @abstractmethod
    async def stream_generate(
        self, *, question: str, contexts: Sequence[str], sources: Sequence[SourceItem]
    ) -> AsyncIterator[str]:
        raise NotImplementedError


class DeepSeekGenerator(Generator):
    def __init__(self, *, model: str, api_key: str, base_url: str) -> None:
        if not api_key:
            raise GeneratorConfigError("缺少 LLM API key。")
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url or None)
        self._model = model

    @staticmethod
    def _build_prompt(*, question: str, contexts: Sequence[str], sources: Sequence[SourceItem]) -> str:
        context_text = "\n\n".join(f"[{idx + 1}] {ctx}" for idx, ctx in enumerate(contexts))
        citations = "\n".join(
            f"- [{idx + 1}] {source.title}" + (f"（{source.url}）" if source.url else "")
            for idx, source in enumerate(sources)
        )
        return (
            "你是企业内部知识库问答助手。请严格基于提供的上下文回答。"
            "如果上下文为空，请明确说明未找到完整答案。"
            "如果上下文非空，请基于上下文总结回答要点；"
            "即使上下文包含运行日志/错误信息，也要提炼其关键结论并回答用户问题。"
            "不要因为上下文不包含问题中的某些字面短语就否定答案；不要编造上下文中没有的信息。"
            "如果用户要求字数（如“大概100字/约100字”），尽量控制在该范围内。"
            "输出必须是 Markdown 格式。"
            "\n\n问题:\n"
            f"{question}\n\n"
            "上下文:\n"
            f"{context_text or '无'}\n\n"
            "参考来源:\n"
            f"{citations or '无'}"
        )

    async def generate(self, *, question: str, contexts: Sequence[str], sources: Sequence[SourceItem]) -> str:
        prompt = self._build_prompt(question=question, contexts=contexts, sources=sources)
        response = await self._client.chat.completions.create(
            model=self._model,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你必须使用中文并严格输出 Markdown。"
                        "回答结构固定为：\n"
                        "## 回答\n"
                        "<正文>\n\n"
                        "## 参考来源\n"
                        "- [1] <来源标题>\n"
                        "- [2] <来源标题>\n"
                        "若无来源，输出：- 无。\n"
                        "不要输出与该结构无关的额外标题。"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or "未生成回答。"

    async def stream_generate(
        self, *, question: str, contexts: Sequence[str], sources: Sequence[SourceItem]
    ) -> AsyncIterator[str]:
        prompt = self._build_prompt(question=question, contexts=contexts, sources=sources)
        stream = await self._client.chat.completions.create(
            model=self._model,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你必须使用中文并严格输出 Markdown。"
                        "回答结构固定为：\n"
                        "## 回答\n"
                        "<正文>\n\n"
                        "## 参考来源\n"
                        "- [1] <来源标题>\n"
                        "- [2] <来源标题>\n"
                        "若无来源，输出：- 无。\n"
                        "不要输出与该结构无关的额外标题。"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )
        async for chunk in stream:
            delta = (chunk.choices[0].delta.content or "") if chunk.choices else ""
            if delta:
                yield delta


class OpenAIGenerator(DeepSeekGenerator):
    pass

