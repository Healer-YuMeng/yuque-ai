from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.core.config import settings
from app.data.yuque_loader import YuqueLoaderError
from app.data.yuque_loader import YuqueLoader
from app.rag.generator import GeneratorConfigError
from app.schemas.docs import DocMeta, DocSuggestRequest, DocSuggestResponse
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    MCPCapabilitiesResponse,
    RebuildIndexResponse,
    RuntimeModeResponse,
)
from app.service.qa_service import QAService


router = APIRouter()


def get_qa_service() -> QAService:
    from app.main import app

    return app.state.qa_service


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/runtime-mode", response_model=RuntimeModeResponse)
async def runtime_mode(qa_service: QAService = Depends(get_qa_service)) -> RuntimeModeResponse:
    mode, label = qa_service.runtime_mode()
    return RuntimeModeResponse(mode=mode, label=label)


@router.get("/mcp/capabilities", response_model=MCPCapabilitiesResponse)
async def mcp_capabilities(qa_service: QAService = Depends(get_qa_service)) -> MCPCapabilitiesResponse:
    return MCPCapabilitiesResponse(**qa_service.mcp_capabilities())


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, qa_service: QAService = Depends(get_qa_service)) -> ChatResponse:
    try:
        return await qa_service.chat(request.question, model=request.model, owner=request.owner)
    except (GeneratorConfigError, YuqueLoaderError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest, qa_service: QAService = Depends(get_qa_service)) -> StreamingResponse:
    async def event_generator():
        try:
            async for item in qa_service.chat_stream(request.question, model=request.model, owner=request.owner):
                yield (
                    f"event: {item['event']}\n"
                    f"data: {json.dumps(item['data'], ensure_ascii=False)}\n\n"
                )
        except (GeneratorConfigError, YuqueLoaderError) as exc:
            payload = {"message": str(exc)}
            yield f"event: error\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/index/rebuild", response_model=RebuildIndexResponse)
async def rebuild_index(qa_service: QAService = Depends(get_qa_service)) -> RebuildIndexResponse:
    docs, chunks = await qa_service.rebuild_index(bootstrap_query="退款")
    return RebuildIndexResponse(indexed_documents=docs, indexed_chunks=chunks)


@router.post("/sync/yuque", response_model=RebuildIndexResponse)
async def sync_yuque(qa_service: QAService = Depends(get_qa_service)) -> RebuildIndexResponse:
    docs, chunks = await qa_service.rebuild_index(bootstrap_query="知识库")
    return RebuildIndexResponse(indexed_documents=docs, indexed_chunks=chunks)


def _compute_yuque_scope(owner: str | None) -> str:
    default_scope = (settings.yuque_scope or "").strip().strip("/")
    if not owner:
        return default_scope
    owner = owner.strip().strip("/")
    if not default_scope or "/" not in default_scope:
        return owner
    _, repo = default_scope.split("/", 1)
    return f"{owner}/{repo}"


@router.post("/docs/suggest", response_model=DocSuggestResponse)
async def docs_suggest(request: DocSuggestRequest) -> DocSuggestResponse:
    scope = _compute_yuque_scope(request.owner)
    loader = YuqueLoader(
        token=settings.yuque_token,
        base_url=settings.yuque_base_url,
        timeout_s=settings.yuque_timeout_s,
        scope=scope,
    )
    q = (request.query or "").strip()

    try:
        docs = await loader.list_docs(book=scope, offset=0, limit=80)
    except YuqueLoaderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        await loader.close()

    # 如果没有关键词（用户只输入了 @），直接返回前 N 个文档作为“目录”入口。
    if not q:
        docs_out = [
            DocMeta(
                id=d.id,
                slug=d.slug,
                title=d.title,
                url=d.url,
                updated_at=d.updated_at,
            )
            for d in docs[:10]
        ]
        return DocSuggestResponse(docs=docs_out)

    # 双向匹配：标题包含/或标题被用户输入包含（适配“@前缀输入”的场景）
    candidates = [
        d
        for d in docs
        if d.title
        and (q in d.title or d.title in q or (d.slug and q in d.slug))
    ]
    # 如果仍然为空，退化为只要“包含任意字符序列”的粗匹配（避免完全没结果）
    if not candidates:
        candidates = [d for d in docs if d.title and d.title.startswith(q[:4])]

    docs_out = [
        DocMeta(
            id=d.id,
            slug=d.slug,
            title=d.title,
            url=d.url,
            updated_at=d.updated_at,
        )
        for d in candidates[:10]
    ]
    return DocSuggestResponse(docs=docs_out)

