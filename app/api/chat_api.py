from __future__ import annotations

import json

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response, StreamingResponse
from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError, RateLimitError

from app.core.config import settings
from app.core.yuque_credentials import (
    default_yuque_scope_for_profile,
    normalize_yuque_token_profile,
    secondary_yuque_configured,
    yuque_token_for_profile,
)
from app.core.logger import get_logger
from app.data.yuque_images import decode_image_proxy_token, is_allowed_yuque_image_url
from app.data.yuque_loader import YuqueLoaderError
from app.data.yuque_loader import YuqueLoader, YuqueTocNode
from app.rag.generator import GeneratorConfigError
from app.schemas.docs import DocMeta, DocSuggestRequest, DocSuggestResponse
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ChatV2Request,
    ChatV2Response,
    ChatV3Request,
    ChatV3Response,
    ChatV3CapabilitiesResponse,
    ChatV4Request,
    ChatV4Response,
    ChatV4CapabilitiesResponse,
    TrialCredentialsRequest,
    TrialCredentialsResponse,
    VisitorTrialApplyRequest,
    VisitorProfileResponse,
    GuideDocTitlesResponse,
    ResetSessionRequest,
    HealthResponse,
    MCPCapabilitiesResponse,
    RebuildIndexResponse,
    RuntimeModeResponse,
)
from app.service.qa_service import QAService


router = APIRouter()
logger = get_logger(__name__)


def _ensure_yuque_token_for_profile(token_profile: str | None) -> str:
    if normalize_yuque_token_profile(token_profile) == "secondary" and not secondary_yuque_configured():
        raise HTTPException(status_code=503, detail="未配置 YUQUE_TOKEN_SECONDARY，无法使用副账号。")
    token = yuque_token_for_profile(token_profile)
    if not token.strip():
        raise HTTPException(status_code=503, detail="语雀 Token 未配置或为空。")
    return token


@router.get("/yuque/asset")
async def yuque_asset_proxy(
    t: str = Query(..., min_length=1, description="base64url 编码的原始语雀媒体 URL"),
    token_profile: str | None = Query(default=None, max_length=24),
) -> Response:
    """同源代理语雀 CDN 图片/视频，供前端展示（带主机白名单）。"""
    try:
        raw_url = decode_image_proxy_token(t)
    except (ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=400, detail="invalid asset token") from exc
    if not is_allowed_yuque_image_url(raw_url):
        raise HTTPException(status_code=400, detail="url host not allowed")
    token = _ensure_yuque_token_for_profile(token_profile)
    headers = {
        "X-Auth-Token": token,
        "User-Agent": "enterprise-rag-mvp/0.1",
        "Referer": "https://www.yuque.com/",
        "Accept": "*/*",
    }
    try:
        async with httpx.AsyncClient(
            timeout=settings.yuque_timeout_s,
            follow_redirects=True,
        ) as client:
            resp = await client.get(raw_url, headers=headers)
    except httpx.HTTPError as exc:
        logger.warning("yuque_asset_proxy_fetch_failed err=%s", exc)
        raise HTTPException(status_code=502, detail="upstream fetch failed") from exc
    if resp.status_code != 200:
        raise HTTPException(
            status_code=404 if resp.status_code == 404 else 502,
            detail=f"upstream status {resp.status_code}",
        )
    ct = (resp.headers.get("content-type") or "application/octet-stream").split(";")[0].strip()
    return Response(content=resp.content, media_type=ct)


_LLM_BUSY_HINT = (
    "当前大模型服务端繁忙（503），请隔几分钟再试；"
    "若持续出现，可在 .env 中临时更换其它兼容 OpenAI 格式的 API 地址或模型。"
)


def _is_llm_service_unavailable(exc: APIError) -> bool:
    if isinstance(exc, APIStatusError) and getattr(exc, "status_code", None) == 503:
        return True
    err_type = (getattr(exc, "type", None) or "").lower()
    if "service_unavailable" in err_type:
        return True
    raw = str(exc).lower()
    return "503" in raw and ("too busy" in raw or "service_unavailable" in raw or "繁忙" in raw)


def _chat_stream_error_message(exc: BaseException) -> str:
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return (
            "无法连接大模型服务（Connection error）。请检查：本机网络、VPN/代理、防火墙，"
            "以及 .env 中 DEEPSEEK_BASE_URL / LLM_BASE_URL 是否指向可访问的地址；"
            "若走系统代理，确认 httpx 能连到该 API。"
        )
    if isinstance(exc, RateLimitError):
        return f"大模型限流：{exc}"
    if isinstance(exc, APIError):
        if _is_llm_service_unavailable(exc):
            return _LLM_BUSY_HINT
        return f"大模型 API 错误：{exc}"
    return str(exc) or "流式问答失败"


def get_qa_service() -> QAService:
    from app.main import app

    return app.state.qa_service


@router.post("/chat/session/reset")
async def reset_chat_session(
    request: ResetSessionRequest,
    qa_service: QAService = Depends(get_qa_service),
) -> dict[str, str]:
    """强制清空某个 session 的服务端历史，保证“新会话从零开始”不串话。"""
    await qa_service.reset_session(session_id=request.session_id, chat_mode=request.chat_mode)
    return {"status": "ok"}


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/runtime-mode", response_model=RuntimeModeResponse)
async def runtime_mode(qa_service: QAService = Depends(get_qa_service)) -> RuntimeModeResponse:
    mode, label = qa_service.runtime_mode()
    return RuntimeModeResponse(mode=mode, label=label, llm_model=settings.llm_model)


@router.get("/mcp/capabilities", response_model=MCPCapabilitiesResponse)
async def mcp_capabilities(qa_service: QAService = Depends(get_qa_service)) -> MCPCapabilitiesResponse:
    data = qa_service.mcp_capabilities()
    pri_login, sec_login = await qa_service.resolve_yuque_token_logins()
    data["yuque_token_primary_login"] = pri_login
    data["yuque_token_secondary_login"] = sec_login
    return MCPCapabilitiesResponse(**data)


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, qa_service: QAService = Depends(get_qa_service)) -> ChatResponse:
    try:
        _ensure_yuque_token_for_profile(request.token_profile)
        return await qa_service.chat(
            request.question,
            model=request.model,
            owner=request.owner,
            selected_yuque_docs=request.selected_yuque_docs,
            token_profile=request.token_profile,
            chat_mode=request.chat_mode,
            session_id=request.session_id,
        )
    except (GeneratorConfigError, YuqueLoaderError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest, qa_service: QAService = Depends(get_qa_service)) -> StreamingResponse:
    async def event_generator():
        try:
            _ensure_yuque_token_for_profile(request.token_profile)
            async for item in qa_service.chat_stream(
                request.question,
                model=request.model,
                owner=request.owner,
                selected_yuque_docs=request.selected_yuque_docs,
                token_profile=request.token_profile,
                chat_mode=request.chat_mode,
                session_id=request.session_id,
            ):
                yield (
                    f"event: {item['event']}\n"
                    f"data: {json.dumps(item['data'], ensure_ascii=False)}\n\n"
                )
        except (GeneratorConfigError, YuqueLoaderError) as exc:
            payload = {"message": str(exc)}
            yield f"event: error\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        except (APIConnectionError, APITimeoutError, APIError, RateLimitError) as exc:
            logger.warning("chat_stream_llm_client_error err=%s", exc)
            payload = {"message": _chat_stream_error_message(exc)}
            yield f"event: error\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        except Exception as exc:
            logger.exception("chat_stream_unhandled")
            payload = {"message": _chat_stream_error_message(exc)}
            yield f"event: error\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/chat/v2", response_model=ChatV2Response)
async def chat_v2(request: ChatV2Request, qa_service: QAService = Depends(get_qa_service)) -> ChatV2Response:
    if not settings.chat_v15_enabled:
        raise HTTPException(status_code=503, detail="V1.5 多媒体链路未开启，请先设置 CHAT_V15_ENABLED=true。")
    try:
        _ensure_yuque_token_for_profile(request.token_profile)
        return await qa_service.chat_v2(
            request.question,
            model=request.model,
            owner=request.owner,
            token_profile=request.token_profile,
            chat_mode=request.chat_mode,
            session_id=request.session_id,
        )
    except (GeneratorConfigError, YuqueLoaderError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/chat/v2/stream")
async def chat_v2_stream(request: ChatV2Request, qa_service: QAService = Depends(get_qa_service)) -> StreamingResponse:
    async def event_generator():
        if not settings.chat_v15_enabled:
            payload = {"message": "V1.5 多媒体链路未开启，请先设置 CHAT_V15_ENABLED=true。"}
            yield f"event: error\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
            return
        try:
            _ensure_yuque_token_for_profile(request.token_profile)
            async for item in qa_service.chat_v2_stream(
                request.question,
                model=request.model,
                owner=request.owner,
                token_profile=request.token_profile,
                chat_mode=request.chat_mode,
                session_id=request.session_id,
            ):
                yield (
                    f"event: {item['event']}\n"
                    f"data: {json.dumps(item['data'], ensure_ascii=False)}\n\n"
                )
        except (GeneratorConfigError, YuqueLoaderError) as exc:
            payload = {"message": str(exc)}
            yield f"event: error\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        except (APIConnectionError, APITimeoutError, APIError, RateLimitError) as exc:
            logger.warning("chat_v2_stream_llm_client_error err=%s", exc)
            payload = {"message": _chat_stream_error_message(exc)}
            yield f"event: error\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        except Exception:
            logger.exception("chat_v2_stream_unhandled")
            payload = {"message": "V1.5 流式问答失败，请稍后重试。"}
            yield f"event: error\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/chat/v2/guide-titles", response_model=GuideDocTitlesResponse)
async def chat_v2_guide_titles(qa_service: QAService = Depends(get_qa_service)) -> GuideDocTitlesResponse:
    data = qa_service.guide_titles_state()
    return GuideDocTitlesResponse(**data)


@router.get("/chat/v3/capabilities", response_model=ChatV3CapabilitiesResponse)
async def chat_v3_capabilities(qa_service: QAService = Depends(get_qa_service)) -> ChatV3CapabilitiesResponse:
    data = qa_service.guide_titles_state()
    return ChatV3CapabilitiesResponse(
        enabled=bool(settings.chat_v3_enabled),
        toc_loaded=bool((data.get("total_nodes") or 0) > 0),
        profile_enabled=True,
    )


@router.post("/chat/v3", response_model=ChatV3Response)
async def chat_v3(request: ChatV3Request, qa_service: QAService = Depends(get_qa_service)) -> ChatV3Response:
    if not settings.chat_v3_enabled:
        raise HTTPException(status_code=503, detail="V3 链路未开启，请先设置 CHAT_V3_ENABLED=true。")
    try:
        _ensure_yuque_token_for_profile(request.token_profile)
        # 非流式：复用 stream 结果拼装（简单实现）
        answer = ""
        async for item in qa_service.chat_v3_stream(
            request.question,
            model=request.model,
            owner=request.owner,
            token_profile=request.token_profile,
            chat_mode=request.chat_mode,
            session_id=request.session_id,
        ):
            if item.get("event") == "token":
                answer += str((item.get("data") or {}).get("token") or "")
            if item.get("event") == "done":
                data = item.get("data") or {}
                if isinstance(data, dict):
                    data["answer"] = answer.strip() or str(data.get("answer") or "")
                    return ChatV3Response(**data)
        return ChatV3Response(answer=answer or "未生成回答。", sources=[], fallback_used=True)
    except (GeneratorConfigError, YuqueLoaderError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/chat/v3/stream")
async def chat_v3_stream(request: ChatV3Request, qa_service: QAService = Depends(get_qa_service)) -> StreamingResponse:
    async def event_generator():
        if not settings.chat_v3_enabled:
            payload = {"message": "V3 链路未开启，请先设置 CHAT_V3_ENABLED=true。"}
            yield f"event: error\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
            return
        try:
            _ensure_yuque_token_for_profile(request.token_profile)
            async for item in qa_service.chat_v3_stream(
                request.question,
                model=request.model,
                owner=request.owner,
                token_profile=request.token_profile,
                chat_mode=request.chat_mode,
                session_id=request.session_id,
            ):
                yield (
                    f"event: {item['event']}\n"
                    f"data: {json.dumps(item['data'], ensure_ascii=False)}\n\n"
                )
        except (GeneratorConfigError, YuqueLoaderError) as exc:
            payload = {"message": str(exc)}
            yield f"event: error\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        except (APIConnectionError, APITimeoutError, APIError, RateLimitError) as exc:
            logger.warning("chat_v3_stream_llm_client_error err=%s", exc)
            payload = {"message": _chat_stream_error_message(exc)}
            yield f"event: error\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        except Exception:
            logger.exception("chat_v3_stream_unhandled")
            payload = {"message": "V3 流式问答失败，请稍后重试。"}
            yield f"event: error\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/chat/v4/capabilities", response_model=ChatV4CapabilitiesResponse)
async def chat_v4_capabilities(qa_service: QAService = Depends(get_qa_service)) -> ChatV4CapabilitiesResponse:
    data = qa_service.guide_titles_state()
    return ChatV4CapabilitiesResponse(
        enabled=bool(settings.chat_v4_enabled),
        toc_loaded=bool((data.get("total_nodes") or 0) > 0),
        catalog_state_enabled=True,
    )


@router.post("/chat/v4", response_model=ChatV4Response)
async def chat_v4(request: ChatV4Request, qa_service: QAService = Depends(get_qa_service)) -> ChatV4Response:
    if not settings.chat_v4_enabled:
        raise HTTPException(status_code=503, detail="V4 链路未开启，请先设置 CHAT_V4_ENABLED=true。")
    try:
        _ensure_yuque_token_for_profile(request.token_profile)
        answer = ""
        async for item in qa_service.chat_v4_stream(
            request.question,
            model=request.model,
            owner=request.owner,
            token_profile=request.token_profile,
            chat_mode=request.chat_mode,
            session_id=request.session_id,
            selected_yuque_docs=request.selected_yuque_docs,
        ):
            if item.get("event") == "token":
                answer += str((item.get("data") or {}).get("token") or "")
            if item.get("event") == "done":
                data = item.get("data") or {}
                if isinstance(data, dict):
                    data["answer"] = answer.strip() or str(data.get("answer") or "")
                    return ChatV4Response(**data)
        return ChatV4Response(answer=answer or "未生成回答。", sources=[], fallback_used=True)
    except (GeneratorConfigError, YuqueLoaderError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/chat/v4/stream")
async def chat_v4_stream(request: ChatV4Request, qa_service: QAService = Depends(get_qa_service)) -> StreamingResponse:
    async def event_generator():
        if not settings.chat_v4_enabled:
            payload = {"message": "V4 链路未开启，请先设置 CHAT_V4_ENABLED=true。"}
            yield f"event: error\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
            return
        try:
            _ensure_yuque_token_for_profile(request.token_profile)
            async for item in qa_service.chat_v4_stream(
                request.question,
                model=request.model,
                owner=request.owner,
                token_profile=request.token_profile,
                chat_mode=request.chat_mode,
                session_id=request.session_id,
                selected_yuque_docs=request.selected_yuque_docs,
            ):
                yield (
                    f"event: {item['event']}\n"
                    f"data: {json.dumps(item['data'], ensure_ascii=False)}\n\n"
                )
        except (GeneratorConfigError, YuqueLoaderError) as exc:
            payload = {"message": str(exc)}
            yield f"event: error\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        except (APIConnectionError, APITimeoutError, APIError, RateLimitError) as exc:
            logger.warning("chat_v4_stream_llm_client_error err=%s", exc)
            payload = {"message": _chat_stream_error_message(exc)}
            yield f"event: error\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        except Exception:
            logger.exception("chat_v4_stream_unhandled")
            payload = {"message": "V4 流式问答失败，请稍后重试。"}
            yield f"event: error\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/chat/v4/trial-credentials", response_model=TrialCredentialsResponse)
async def chat_v4_trial_credentials(
    request: TrialCredentialsRequest,
    qa_service: QAService = Depends(get_qa_service),
) -> TrialCredentialsResponse:
    if not settings.chat_v4_enabled:
        raise HTTPException(status_code=503, detail="V4 链路未开启。")
    return await qa_service.issue_v4_trial_credentials(session_id=request.session_id)


@router.post("/visitor/trial/apply", response_model=TrialCredentialsResponse)
async def visitor_trial_apply(
    request: VisitorTrialApplyRequest,
    qa_service: QAService = Depends(get_qa_service),
) -> TrialCredentialsResponse:
    return await qa_service.apply_visitor_trial_account(
        session_id=request.session_id,
        name=request.name,
        org_name=request.org_name,
        contact=request.contact,
        interested_product=request.interested_product,
        concern=request.concern,
    )


@router.get("/visitor/profile", response_model=VisitorProfileResponse)
async def visitor_profile(
    session_id: str = Query(..., min_length=1, max_length=120),
    qa_service: QAService = Depends(get_qa_service),
) -> VisitorProfileResponse:
    return await qa_service.visitor_profile_summary(session_id=session_id)


@router.post("/index/rebuild", response_model=RebuildIndexResponse)
async def rebuild_index(qa_service: QAService = Depends(get_qa_service)) -> RebuildIndexResponse:
    docs, chunks = await qa_service.rebuild_index(bootstrap_query="退款")
    return RebuildIndexResponse(indexed_documents=docs, indexed_chunks=chunks)


@router.post("/sync/yuque", response_model=RebuildIndexResponse)
async def sync_yuque(qa_service: QAService = Depends(get_qa_service)) -> RebuildIndexResponse:
    docs, chunks = await qa_service.rebuild_index(bootstrap_query="知识库")
    return RebuildIndexResponse(indexed_documents=docs, indexed_chunks=chunks)


def _compute_yuque_scope(owner: str | None, token_profile: str | None = None) -> str:
    default_scope = default_yuque_scope_for_profile(token_profile).strip().strip("/")
    if not owner:
        return default_scope
    owner = owner.strip().strip("/")
    if not default_scope or "/" not in default_scope:
        return owner
    _, repo = default_scope.split("/", 1)
    return f"{owner}/{repo}"


def _slug_tail_from_yuque_url(url: str) -> str:
    u = (url or "").strip().rstrip("/")
    if not u:
        return ""
    parts = [p for p in u.split("/") if p]
    return parts[-1] if parts else ""


def _toc_node_to_doc_meta(node: YuqueTocNode) -> DocMeta:
    is_doc = node.doc_id is not None
    kind = "doc" if is_doc else "title"
    slug_tail = _slug_tail_from_yuque_url(node.url)
    return DocMeta(
        id=node.doc_id,
        slug=slug_tail or None,
        title=node.title or ("(未命名)" if is_doc else "（分组）"),
        url=node.url or None,
        updated_at=None,
        toc_uuid=node.uuid or None,
        toc_parent_uuid=node.parent_uuid or None,
        toc_level=node.level,
        toc_kind=kind,
        toc_selectable=is_doc,
    )


def _toc_nodes_for_query(nodes: list[YuqueTocNode], q: str) -> list[YuqueTocNode]:
    """按关键词筛选：保留匹配节点及其祖先，以维持目录树上下文。"""
    qn = q.strip().lower()
    if not qn:
        return list(nodes)
    by_uuid: dict[str, YuqueTocNode] = {}
    for n in nodes:
        if n.uuid:
            by_uuid[n.uuid] = n
    keep: set[str] = set()
    for n in nodes:
        if not n.uuid:
            continue
        title_l = (n.title or "").lower()
        slug_l = _slug_tail_from_yuque_url(n.url).lower()
        if qn in title_l or (slug_l and qn in slug_l):
            keep.add(n.uuid)
            parent = n.parent_uuid
            while parent and parent in by_uuid:
                keep.add(parent)
                parent = by_uuid[parent].parent_uuid
    return [n for n in nodes if n.uuid in keep]


async def _fetch_toc_doc_metas(owner: str | None, token_profile: str | None = None) -> list[DocMeta]:
    """拉取当前作用域下知识库目录：优先 TOC；失败或为空时用文档列表兜底（与 /docs/suggest 一致）。"""
    scope = _compute_yuque_scope(owner, token_profile=token_profile)
    loader = YuqueLoader(
        token=yuque_token_for_profile(token_profile),
        base_url=settings.yuque_base_url,
        timeout_s=settings.yuque_timeout_s,
        scope=scope,
    )
    toc_nodes: list[YuqueTocNode] = []
    docs: list = []
    try:
        try:
            toc_nodes = await loader.get_book_toc(book=scope)
        except YuqueLoaderError:
            toc_nodes = []
        if toc_nodes:
            return [_toc_node_to_doc_meta(n) for n in toc_nodes[:500]]
        docs = await loader.list_docs(book=scope, offset=0, limit=100)
        return [
            DocMeta(
                id=d.id,
                slug=d.slug or None,
                title=d.title,
                url=d.url or None,
                updated_at=d.updated_at,
            )
            for d in docs
        ]
    finally:
        await loader.close()


def _list_docs_fallback_metas(docs: list, q: str, *, empty_limit: int = 10, match_limit: int = 10) -> list[DocMeta]:
    if not q:
        return [
            DocMeta(id=d.id, slug=d.slug, title=d.title, url=d.url, updated_at=d.updated_at)
            for d in docs[:empty_limit]
        ]
    candidates = [
        d
        for d in docs
        if d.title
        and (q in d.title or d.title in q or (d.slug and q in d.slug))
    ]
    if not candidates:
        candidates = [d for d in docs if d.title and d.title.startswith(q[:4])]
    return [
        DocMeta(id=d.id, slug=d.slug, title=d.title, url=d.url, updated_at=d.updated_at)
        for d in candidates[:match_limit]
    ]


@router.get("/docs/toc", response_model=DocSuggestResponse)
async def docs_toc(
    owner: str | None = Query(default=None, max_length=120),
    token_profile: str | None = Query(default=None, max_length=24),
) -> DocSuggestResponse:
    """返回当前知识库目录树（TOC），供前端「知识库列表」面板展示。"""
    try:
        _ensure_yuque_token_for_profile(token_profile)
        docs_out = await _fetch_toc_doc_metas(owner, token_profile=token_profile)
        return DocSuggestResponse(docs=docs_out)
    except YuqueLoaderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/docs/suggest", response_model=DocSuggestResponse)
async def docs_suggest(request: DocSuggestRequest) -> DocSuggestResponse:
    _ensure_yuque_token_for_profile(request.token_profile)
    scope = _compute_yuque_scope(request.owner, token_profile=request.token_profile)
    loader = YuqueLoader(
        token=yuque_token_for_profile(request.token_profile),
        base_url=settings.yuque_base_url,
        timeout_s=settings.yuque_timeout_s,
        scope=scope,
    )
    q = (request.query or "").strip()
    q_lower = q.lower()

    toc_nodes: list[YuqueTocNode] = []
    docs: list = []
    try:
        try:
            toc_nodes = await loader.get_book_toc(book=scope)
        except YuqueLoaderError:
            toc_nodes = []
        try:
            docs = await loader.list_docs(book=scope, offset=0, limit=100)
        except YuqueLoaderError as exc:
            if not toc_nodes:
                raise
            docs = []
    except YuqueLoaderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        await loader.close()

    if toc_nodes:
        if not q_lower:
            docs_out = [_toc_node_to_doc_meta(n) for n in toc_nodes[:500]]
            return DocSuggestResponse(docs=docs_out)
        filtered = _toc_nodes_for_query(toc_nodes, q_lower)
        if filtered:
            docs_out = [_toc_node_to_doc_meta(n) for n in filtered[:120]]
            return DocSuggestResponse(docs=docs_out)

    docs_out = _list_docs_fallback_metas(docs, q, empty_limit=10, match_limit=10)
    return DocSuggestResponse(docs=docs_out)
