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
    _BASE_SYSTEM = (
        "你必须使用中文并严格输出 Markdown。"
        "回答结构固定为：\n"
        "## 回答\n"
        "<正文>\n\n"
        "## 参考来源\n"
        "- [1] <来源标题>\n"
        "- [2] <来源标题>\n"
        "若无来源，输出：- 无。\n"
        "不要输出与该结构无关的额外标题。"
    )

    def __init__(self, *, model: str, api_key: str, base_url: str) -> None:
        if not api_key:
            raise GeneratorConfigError("缺少 LLM API key。")
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url or None)
        self._model = model

    @classmethod
    def _system_message(cls, contexts: Sequence[str]) -> str:
        extra = ""
        if contexts and str(contexts[0]).startswith("【合并知识库清单"):
            extra = (
                "\n当前为「目录树 + 文档清单」合并上下文：在 ## 回答 的正文里，"
                "必须先按层级完整输出目录树（与上下文中 level/缩进一致），"
                "再输出含字数/图片数/可见性等列的 Markdown 表格；表中无数据的格填「未提供」，禁止臆造数字。"
            )
        elif any(str(c).startswith("【文档插图") for c in contexts):
            extra = (
                "\n上下文中含「文档插图」块：其中列出了须在 ## 回答 中插入的 Markdown 图片语法"
                "（路径形如 `/yuque/asset?t=...`），可能附带多模态识读摘要。**必须原样使用**这些 `![...](...)` 行，"
                "勿改写 URL 或查询参数；在正文合适位置插入，并可与块内说明文字配合组织段落。"
            )
        return cls._BASE_SYSTEM + extra

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
                {"role": "system", "content": self._system_message(contexts)},
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
                {"role": "system", "content": self._system_message(contexts)},
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

