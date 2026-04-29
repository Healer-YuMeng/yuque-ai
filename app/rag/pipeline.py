from __future__ import annotations

from typing import Any, AsyncIterator, Dict

from app.rag.generator import Generator, GeneratorConfigError
from app.rag.retriever import RetrievalResult, Retriever
from app.schemas.chat import ChatResponse


class RAGPipeline:
    def __init__(self, *, retriever: Retriever, generator: Generator | None) -> None:
        self._retriever = retriever
        self._generator = generator

    async def run(
        self,
        question: str,
        *,
        retrieval_question: str | None = None,
        generation_question: str | None = None,
        skill_id: str | None = None,
    ) -> ChatResponse:
        retrieval_question = retrieval_question or question
        generation_question = generation_question or question
        if self._generator is None:
            raise GeneratorConfigError("缺少 LLM API key，请在 .env 中配置 DEEPSEEK_API_KEY 或 LLM_API_KEY。")
        retrieval = await self._retriever.retrieve(retrieval_question, skill_id=skill_id)
        answer = await self._generator.generate(
            question=generation_question,
            contexts=retrieval.contexts,
            sources=retrieval.sources,
        )
        debug = retrieval.debug or {
            "retrieval_mode": "yuque_or_vector",
            "fallback_used": retrieval.fallback_used,
            "source_types": sorted({source.source_type for source in retrieval.sources}),
            "source_count": len(retrieval.sources),
        }
        return ChatResponse(
            answer=answer,
            sources=retrieval.sources,
            fallback_used=retrieval.fallback_used,
            debug=debug,
        )

    async def run_stream(
        self,
        question: str,
        *,
        retrieval_question: str | None = None,
        generation_question: str | None = None,
        skill_id: str | None = None,
    ) -> tuple[RetrievalResult, Dict[str, Any], AsyncIterator[str]]:
        retrieval_question = retrieval_question or question
        generation_question = generation_question or question
        if self._generator is None:
            raise GeneratorConfigError("缺少 LLM API key，请在 .env 中配置 DEEPSEEK_API_KEY 或 LLM_API_KEY。")
        retrieval = await self._retriever.retrieve(retrieval_question, skill_id=skill_id)
        debug = retrieval.debug or {
            "retrieval_mode": "yuque_or_vector",
            "fallback_used": retrieval.fallback_used,
            "source_types": sorted({source.source_type for source in retrieval.sources}),
            "source_count": len(retrieval.sources),
        }
        stream = self._generator.stream_generate(
            question=generation_question,
            contexts=retrieval.contexts,
            sources=retrieval.sources,
        )
        return retrieval, debug, stream

