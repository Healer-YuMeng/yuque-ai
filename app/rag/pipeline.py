from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from app.rag.doc_image_enrichment import enrich_retrieval_with_doc_images
from app.rag.generator import Generator, GeneratorConfigError
from app.rag.retriever import RetrievalResult, Retriever
from app.rag.meta_router import should_use_direct_assistant_help
from app.rag.scope_help import apply_scope_help_if_needed, direct_scope_help_retrieval
from app.schemas.chat import ChatResponse


class RAGPipeline:
    def __init__(self, *, retriever: Retriever, generator: Generator | None) -> None:
        self._retriever = retriever
        self._generator = generator

    def _merge_debug(self, retrieval: RetrievalResult) -> Dict[str, Any]:
        defaults = {
            "retrieval_mode": "yuque_or_vector",
            "fallback_used": retrieval.fallback_used,
            "source_types": sorted({source.source_type for source in retrieval.sources}),
            "source_count": len(retrieval.sources),
        }
        return {**defaults, **(retrieval.debug or {})}

    async def retrieve_context(
        self,
        question: str,
        *,
        retrieval_question: str | None = None,
        generation_question: str | None = None,
        skill_id: Optional[str] = None,
        doc_anchors: Optional[List[tuple[int, Optional[str]]]] = None,
    ) -> Tuple[RetrievalResult, Dict[str, Any]]:
        """检索 + scope_help + 可选插图识读；供流式首段 SSE 在生成前插入「识图」阶段。"""
        retrieval_question = retrieval_question or question
        generation_question = generation_question or question
        if self._generator is None:
            raise GeneratorConfigError("缺少 LLM API key，请在 .env 中配置 DEEPSEEK_API_KEY 或 LLM_API_KEY。")
        use_help, help_reason = await should_use_direct_assistant_help(retrieval_question)
        if use_help:
            route = "rule" if help_reason == "rule" else "llm"
            retrieval = direct_scope_help_retrieval(route=route)
        else:
            retrieval = await self._retriever.retrieve(
                retrieval_question, skill_id=skill_id, doc_anchors=doc_anchors
            )
            retrieval = apply_scope_help_if_needed(retrieval, retrieval_question)
            loader = self._retriever.yuque_loader
            if loader is not None:
                retrieval = await enrich_retrieval_with_doc_images(
                    retrieval=retrieval,
                    question=generation_question,
                    loader=loader,
                )
        return retrieval, self._merge_debug(retrieval)

    def stream_answer_tokens(
        self,
        retrieval: RetrievalResult,
        *,
        generation_question: str,
        visitor_sales: bool = False,
    ) -> AsyncIterator[str]:
        if self._generator is None:
            raise GeneratorConfigError("缺少 LLM API key，请在 .env 中配置 DEEPSEEK_API_KEY 或 LLM_API_KEY。")
        return self._generator.stream_generate(
            question=generation_question,
            contexts=retrieval.contexts,
            sources=retrieval.sources,
            visitor_sales=visitor_sales,
        )

    async def run(
        self,
        question: str,
        *,
        retrieval_question: str | None = None,
        generation_question: str | None = None,
        skill_id: str | None = None,
        doc_anchors: Optional[List[tuple[int, Optional[str]]]] = None,
        visitor_sales: bool = False,
    ) -> ChatResponse:
        retrieval_question = retrieval_question or question
        generation_question = generation_question or question
        retrieval, debug = await self.retrieve_context(
            question,
            retrieval_question=retrieval_question,
            generation_question=generation_question,
            skill_id=skill_id,
            doc_anchors=doc_anchors,
        )
        answer = await self._generator.generate(
            question=generation_question,
            contexts=retrieval.contexts,
            sources=retrieval.sources,
            visitor_sales=visitor_sales,
        )
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
        doc_anchors: Optional[List[tuple[int, Optional[str]]]] = None,
        visitor_sales: bool = False,
    ) -> tuple[RetrievalResult, Dict[str, Any], AsyncIterator[str]]:
        retrieval_question = retrieval_question or question
        generation_question = generation_question or question
        retrieval, debug = await self.retrieve_context(
            question,
            retrieval_question=retrieval_question,
            generation_question=generation_question,
            skill_id=skill_id,
            doc_anchors=doc_anchors,
        )
        stream = self.stream_answer_tokens(retrieval, generation_question=generation_question, visitor_sales=visitor_sales)
        return retrieval, debug, stream

