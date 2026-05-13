from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from app.rag.pipeline import RAGPipeline
from app.rag.retriever import RetrievalResult
from app.schemas.chat import SourceItem


class FakeRetriever:
    yuque_loader = None

    def __init__(self) -> None:
        self.last_question: Optional[str] = None
        self.last_skill_id: Optional[str] = None

    async def retrieve(self, question: str, *, skill_id: Optional[str] = None, doc_anchors=None) -> RetrievalResult:
        self.last_question = question
        self.last_skill_id = skill_id
        return RetrievalResult(
            contexts=["ctx"],
            sources=[SourceItem(title="S", url=None, source_type="vector")],
            fallback_used=False,
            debug={"retrieval": True},
        )


class FakeGenerator:
    def __init__(self) -> None:
        self.last_question: Optional[str] = None

    async def generate(
        self,
        *,
        question: str,
        contexts: List[str],
        sources: List[SourceItem],
        visitor_sales: bool = False,
    ) -> str:
        self.last_question = question
        return "answer-ok"

    async def stream_generate(
        self,
        *,
        question: str,
        contexts: List[str],
        sources: List[SourceItem],
        visitor_sales: bool = False,
    ):
        raise NotImplementedError


@pytest.mark.asyncio
async def test_pipeline_uses_generation_question_for_generator() -> None:
    retriever = FakeRetriever()
    generator = FakeGenerator()
    pipeline = RAGPipeline(retriever=retriever, generator=generator)

    resp = await pipeline.run(
        "orig-question",
        retrieval_question="retrieval-question",
        generation_question="generation-question",
        skill_id="smart-summary",
    )
    assert resp.answer == "answer-ok"
    assert retriever.last_question == "retrieval-question"
    assert generator.last_question == "generation-question"

