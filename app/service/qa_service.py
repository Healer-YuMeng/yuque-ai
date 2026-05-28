from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from app.core.config import settings
from app.core.yuque_credentials import (
    normalize_yuque_token_profile,
    secondary_yuque_configured,
    default_yuque_scope_for_profile,
    yuque_token_for_profile,
)
from app.data.mcp_client import YuqueMCPClient
from app.data.splitter import RecursiveTextSplitter, TextChunk
from app.data.yuque_loader import YuqueDocument, YuqueLoader, YuqueLoaderError
from app.conversation.contact_extractor import extract_contact
from app.conversation.lead_nudge_policy import LeadNudgePolicy
from app.conversation.trial_account_pool import allocate_trial_account, load_trial_accounts
from app.conversation.v4_lead_outreach import V4LeadOutreach
from app.conversation.visitor_prompt import build_visitor_generation_question
from app.conversation.visitor_profile import detect_visitor_type
from app.db.repositories import ChatMessageRow, ChatSessionRepository, DocumentRepository, LeadCaptureRepository, QALogRepository
from app.db.profile_repository import ChatSessionProfileRepository
from app.rag.embedder import BGESmallEmbedder, Embedder, OpenAIEmbedder
from app.rag.generator import DeepSeekGenerator, Generator, GeneratorConfigError, OpenAIGenerator
from app.rag.skill_router import route_skill
from app.rag.pipeline import RAGPipeline
from app.rag.retriever import Retriever
from app.schemas.chat import ChatMediaBundle, ChatResponse, ChatV2Response, SelectedYuqueDocRef, TrialCredentialsResponse
from app.service.media_answer_orchestrator import MediaAnswerOrchestrator
from app.service.sales_dialog_orchestrator_v3 import SalesDialogOrchestratorV3
from app.service.sales_dialog_orchestrator_v4 import SalesDialogOrchestratorV4, _strip_media_urls_from_text
from app.storage.vector_store import StoredChunk, VectorStore
from app.core.logger import get_logger

logger = get_logger(__name__)


def _doc_anchor_pairs(selected: Optional[List[SelectedYuqueDocRef]]) -> Optional[List[Tuple[int, Optional[str]]]]:
    if not selected:
        return None
    out: List[Tuple[int, Optional[str]]] = []
    for item in selected:
        slug = (item.slug or "").strip() or None
        out.append((item.doc_id, slug))
    return out


def _sse_stage(stage: str, detail: str, **extra: Any) -> dict[str, Any]:
    """流式 SSE：在首 token 前推送阶段，便于前端展示进度。"""
    payload: dict[str, Any] = {"stage": stage, "detail": detail}
    for key, value in extra.items():
        if value is not None and value != "":
            payload[key] = value
    return {"event": "stage", "data": payload}


def _retrieving_stage_detail(*, runtime_label: str, skill_id: Optional[str], scope: Optional[str] = None) -> str:
    parts = [f"正在检索知识库（{runtime_label}）"]
    if skill_id:
        parts.append(f"技能 `{skill_id}`")
    if scope:
        parts.append(f"作用域 `{scope}`")
    return "，".join(parts) + "…"


def _generating_stage_detail(debug: Optional[Dict[str, Any]]) -> str:
    mode = (debug or {}).get("retrieval_mode") or ""
    if mode == "scope_help_direct":
        return "正在根据内置「助手能力与范围」说明生成回答…"
    if mode == "stale_detector":
        return "正在根据文档列表元信息生成回答…"
    if mode == "mcp_fallback":
        return "正在根据 MCP / 回退检索到的上下文生成回答…"
    return "正在调用大模型生成回答…"


def _vision_stage_detail(debug: Optional[Dict[str, Any]]) -> str:
    n = (debug or {}).get("vision_images_used")
    if isinstance(n, int) and n > 0:
        return f"已完成 {n} 张文档插图的识读，正在生成回答…"
    return "已完成文档插图识读，正在生成回答…"


def _is_visitor_sales(chat_mode: Optional[str]) -> bool:
    return (chat_mode or "visitor_sales") == "visitor_sales"


class QAService:
    def __init__(
        self,
        *,
        yuque_loader: YuqueLoader,
        vector_store: VectorStore,
        document_repository: DocumentRepository,
        qa_log_repository: QALogRepository,
        lead_capture_repository: LeadCaptureRepository,
        chat_session_repository: ChatSessionRepository,
        chat_session_profile_repository: ChatSessionProfileRepository,
    ) -> None:
        self._yuque_loader = yuque_loader
        self._vector_store = vector_store
        self._document_repository = document_repository
        self._qa_log_repository = qa_log_repository
        self._lead_capture_repository = lead_capture_repository
        self._chat_session_repository = chat_session_repository
        self._splitter = RecursiveTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        self._embedder = self._build_embedder()
        self._generator = self._build_generator()
        self._mcp_client = YuqueMCPClient(
            command=settings.mcp_server_command,
            args=settings.mcp_server_args,
            repo_id=settings.yuque_scope,
            search_tool=settings.mcp_search_tool,
            get_doc_tool=settings.mcp_get_doc_tool,
        )
        self._pipeline = RAGPipeline(
            retriever=Retriever(
                vector_store=vector_store,
                embedder=self._embedder,
                mcp_client=self._mcp_client,
                yuque_loader=self._yuque_loader,
                top_k=settings.top_k,
                score_threshold=settings.retrieval_score_threshold,
            ),
            generator=self._generator,
        )
        self._guide_doc_titles: List[str] = []
        self._guide_toc_nodes: List[Dict[str, Any]] = []
        self._guide_titles_refreshed_at: float = 0.0
        self._guide_titles_refresh_lock = asyncio.Lock()
        self._guide_titles_refresh_task: Optional[asyncio.Task[None]] = None
        self._lead_nudge_policy = LeadNudgePolicy(
            rounds_threshold=settings.chat_v15_lead_nudge_rounds,
            stay_seconds_threshold=settings.chat_v15_lead_nudge_stay_s,
        )
        self._chat_session_profile_repository = chat_session_profile_repository

    async def startup(self) -> None:
        await self._document_repository.init_db()
        # 最小实现：启动时做一次过期清理，避免数据无限增长（默认 7 天）。
        await self._chat_session_repository.prune_older_than_days(retention_days=7)
        await self._refresh_guide_doc_titles_if_stale(force=True)
        self._start_guide_titles_refresh_loop()

    async def shutdown(self) -> None:
        await self._stop_guide_titles_refresh_loop()
        await self._yuque_loader.close()

    async def chat(
        self,
        question: str,
        *,
        model: Optional[str] = None,
        owner: Optional[str] = None,
        selected_yuque_docs: Optional[List[SelectedYuqueDocRef]] = None,
        token_profile: Optional[str] = None,
        chat_mode: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> ChatResponse:
        visitor = _is_visitor_sales(chat_mode)
        sid = (session_id or "").strip()
        if sid:
            await self._chat_session_repository.ensure_session(
                session_id=sid, chat_mode=(chat_mode or "visitor_sales"), advisor_role="sales"
            )
            await self._chat_session_repository.append_message(session_id=sid, role="user", content=question)
            vt = detect_visitor_type(question)
            if vt != "unknown":
                await self._chat_session_repository.update_visitor_type(session_id=sid, visitor_type=vt)

        history = await self._history_block_for_session(sid)
        if visitor:
            skill_route = None
            skill_id: Optional[str] = None
            generation_question = self._with_history(build_visitor_generation_question(question), history)
        else:
            skill_route = route_skill(question)
            skill_id = skill_route.skill_id if skill_route else None
            base_q = (
                f"[skill_id={skill_id}]\n{skill_route.generation_instruction}\n\n用户问题：{question}"
                if skill_route
                else question
            )
            generation_question = self._with_history(base_q, history)
        anchors = _doc_anchor_pairs(selected_yuque_docs)
        if model is None and owner is None and normalize_yuque_token_profile(token_profile) == "secondary":
            if not secondary_yuque_configured():
                raise YuqueLoaderError("未配置 YUQUE_TOKEN_SECONDARY，无法使用副账号。")
            return await self._run_one(
                question,
                model=None,
                owner=None,
                skill_id=skill_id,
                generation_question=generation_question,
                selected_yuque_docs=selected_yuque_docs,
                token_profile=token_profile,
                chat_mode=chat_mode,
                session_id=session_id,
            )
        if model is None and owner is None:
            mode, label = self.runtime_mode()
            logger.info("chat_received mode=%s label=%s question=%r", mode, label, question)
            response = await self._pipeline.run(
                question,
                retrieval_question=question,
                generation_question=generation_question,
                skill_id=skill_id,
                doc_anchors=anchors,
                visitor_sales=visitor,
            )
            if not settings.expose_source_urls:
                for source in response.sources:
                    source.url = None
            if response.debug is not None and skill_id and skill_route:
                response.debug["skill_id"] = skill_id
                response.debug["skill_instruction"] = skill_route.generation_instruction
            logger.info(
                "chat_completed sources=%d fallback_used=%s",
                len(response.sources),
                response.fallback_used,
            )
            await self._qa_log_repository.log_chat(question=question, response=response)
            if visitor:
                response = await self._apply_visitor_sales_client_mask(
                    question=question, session_id=session_id, response=response
                )
            if sid:
                await self._chat_session_repository.append_message(
                    session_id=sid, role="assistant", content=response.answer
                )
            return response

        response = await self._run_one(
            question,
            model=model,
            owner=owner,
            skill_id=skill_id,
            generation_question=generation_question,
            selected_yuque_docs=selected_yuque_docs,
            token_profile=token_profile,
            chat_mode=chat_mode,
            session_id=session_id,
        )
        if sid:
            await self._chat_session_repository.append_message(session_id=sid, role="assistant", content=response.answer)
        return response

    async def chat_stream(
        self,
        question: str,
        *,
        model: Optional[str] = None,
        owner: Optional[str] = None,
        selected_yuque_docs: Optional[List[SelectedYuqueDocRef]] = None,
        token_profile: Optional[str] = None,
        chat_mode: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> AsyncIterator[dict[str, Any]]:
        visitor = _is_visitor_sales(chat_mode)
        sid = (session_id or "").strip()
        if sid:
            await self._chat_session_repository.ensure_session(
                session_id=sid, chat_mode=(chat_mode or "visitor_sales"), advisor_role="sales"
            )
            await self._chat_session_repository.append_message(session_id=sid, role="user", content=question)
            vt = detect_visitor_type(question)
            if vt != "unknown":
                await self._chat_session_repository.update_visitor_type(session_id=sid, visitor_type=vt)

        history = await self._history_block_for_session(sid)
        if visitor:
            skill_route = None
            skill_id: Optional[str] = None
            generation_question = self._with_history(build_visitor_generation_question(question), history)
        else:
            skill_route = route_skill(question)
            skill_id = skill_route.skill_id if skill_route else None
            base_q = (
                f"[skill_id={skill_id}]\n{skill_route.generation_instruction}\n\n用户问题：{question}"
                if skill_route
                else question
            )
            generation_question = self._with_history(base_q, history)
        anchors = _doc_anchor_pairs(selected_yuque_docs)
        if model is None and owner is None and normalize_yuque_token_profile(token_profile) == "secondary":
            if not secondary_yuque_configured():
                raise YuqueLoaderError("未配置 YUQUE_TOKEN_SECONDARY，无法使用副账号。")
            async for event in self._run_one_stream(
                question,
                model=None,
                owner=None,
                skill_id=skill_id,
                generation_question=generation_question,
                selected_yuque_docs=selected_yuque_docs,
                token_profile=token_profile,
                chat_mode=chat_mode,
                session_id=session_id,
            ):
                yield event
            return
        if model is None and owner is None:
            mode, label = self.runtime_mode()
            logger.info("chat_received mode=%s label=%s question=%r", mode, label, question)
            yield _sse_stage(
                "retrieving",
                _retrieving_stage_detail(runtime_label=label, skill_id=skill_id),
                mode=mode,
                skill_id=skill_id,
            )
            retrieval, debug = await self._pipeline.retrieve_context(
                question,
                retrieval_question=question,
                generation_question=generation_question,
                skill_id=skill_id,
                doc_anchors=anchors,
            )
            if (debug or {}).get("vision_images_used"):
                yield _sse_stage(
                    "vision",
                    _vision_stage_detail(debug),
                    vision_images_used=(debug or {}).get("vision_images_used"),
                    vision_model=(debug or {}).get("vision_model"),
                )
            yield _sse_stage(
                "generating",
                _generating_stage_detail(debug),
                retrieval_mode=(debug or {}).get("retrieval_mode"),
            )
            answer_stream = self._pipeline.stream_answer_tokens(
                retrieval, generation_question=generation_question, visitor_sales=visitor
            )
            answer_parts: List[str] = []
            async for token in answer_stream:
                answer_parts.append(token)
                yield {"event": "token", "data": {"token": token}}

            answer = "".join(answer_parts) or "未生成回答。"
            response = ChatResponse(
                answer=answer,
                sources=retrieval.sources,
                fallback_used=retrieval.fallback_used,
                debug=debug,
            )
            if not settings.expose_source_urls:
                for source in response.sources:
                    source.url = None
            if response.debug is not None and skill_id and skill_route:
                response.debug["skill_id"] = skill_id
            logger.info(
                "chat_completed sources=%d fallback_used=%s",
                len(response.sources),
                response.fallback_used,
            )
            await self._qa_log_repository.log_chat(question=question, response=response)
            if visitor:
                response = await self._apply_visitor_sales_client_mask(
                    question=question, session_id=session_id, response=response
                )
            if sid:
                await self._chat_session_repository.append_message(
                    session_id=sid, role="assistant", content=response.answer
                )
            yield {"event": "done", "data": response.model_dump()}
            return

        async for event in self._run_one_stream(
            question,
            model=model,
            owner=owner,
            skill_id=skill_id,
            generation_question=generation_question,
            selected_yuque_docs=selected_yuque_docs,
            token_profile=token_profile,
            chat_mode=chat_mode,
            session_id=session_id,
        ):
            if event.get("event") == "done" and sid:
                try:
                    data = event.get("data") or {}
                    answer = str((data.get("answer") if isinstance(data, dict) else "") or "")
                    if answer.strip():
                        await self._chat_session_repository.append_message(
                            session_id=sid, role="assistant", content=answer
                        )
                except Exception:
                    pass
            yield event

    async def chat_v2(
        self,
        question: str,
        *,
        model: Optional[str] = None,
        owner: Optional[str] = None,
        token_profile: Optional[str] = None,
        chat_mode: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> ChatV2Response:
        await self._refresh_guide_doc_titles_if_stale()
        sid = (session_id or "").strip()
        if sid:
            await self._chat_session_repository.ensure_session(
                session_id=sid, chat_mode=(chat_mode or "visitor_sales"), advisor_role="sales"
            )
            await self._chat_session_repository.append_message(session_id=sid, role="user", content=question)
            vt = detect_visitor_type(question)
            if vt != "unknown":
                await self._chat_session_repository.update_visitor_type(session_id=sid, visitor_type=vt)

        scope = self._compute_yuque_scope(owner, token_profile)
        generator = self._build_generator_by_selected_model(model or settings.llm_model)
        mcp_client = self._build_mcp_client(scope)
        orchestrator = self._build_media_orchestrator(mcp_client=mcp_client, generator=generator)

        skill_route = route_skill(question)
        skill_instruction = (
            skill_route.generation_instruction if skill_route else self._auto_skill_instruction_for_v2(question)
        )
        response = await orchestrator.answer(
            question=question,
            session_id=sid or None,
            skill_instruction=skill_instruction,
        )
        if sid:
            await self._chat_session_repository.append_message(
                session_id=sid, role="assistant", content=response.answer
            )
        return response

    async def chat_v2_stream(
        self,
        question: str,
        *,
        model: Optional[str] = None,
        owner: Optional[str] = None,
        token_profile: Optional[str] = None,
        chat_mode: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> AsyncIterator[dict[str, Any]]:
        await self._refresh_guide_doc_titles_if_stale()
        sid = (session_id or "").strip()
        if sid:
            await self._chat_session_repository.ensure_session(
                session_id=sid, chat_mode=(chat_mode or "visitor_sales"), advisor_role="sales"
            )
            await self._chat_session_repository.append_message(session_id=sid, role="user", content=question)
            vt = detect_visitor_type(question)
            if vt != "unknown":
                await self._chat_session_repository.update_visitor_type(session_id=sid, visitor_type=vt)

        scope = self._compute_yuque_scope(owner, token_profile)
        generator = self._build_generator_by_selected_model(model or settings.llm_model)
        mcp_client = self._build_mcp_client(scope)
        orchestrator = self._build_media_orchestrator(mcp_client=mcp_client, generator=generator)

        skill_route = route_skill(question)
        skill_instruction = (
            skill_route.generation_instruction if skill_route else self._auto_skill_instruction_for_v2(question)
        )
        yield _sse_stage("retrieving", "正在检索知识库并聚合多媒体上下文…", mode="mcp_v15")
        yield _sse_stage("generating", "正在流式生成回答…", mode="mcp_v15")
        async for event in orchestrator.answer_stream(
            question=question,
            session_id=sid or None,
            skill_instruction=skill_instruction,
        ):
            if event.get("event") == "done" and sid:
                try:
                    data = event.get("data") or {}
                    answer = str((data.get("answer") if isinstance(data, dict) else "") or "")
                    if answer.strip():
                        await self._chat_session_repository.append_message(
                            session_id=sid, role="assistant", content=answer
                        )
                except Exception:
                    pass
            yield event

    async def chat_v3_stream(
        self,
        question: str,
        *,
        model: Optional[str] = None,
        owner: Optional[str] = None,
        token_profile: Optional[str] = None,
        chat_mode: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """V3：会话画像 + 兴趣推荐 + follow-up 优先答复（SSE）。"""
        await self._refresh_guide_doc_titles_if_stale()
        sid = (session_id or "").strip()
        if sid:
            await self._chat_session_repository.ensure_session(
                session_id=sid, chat_mode=(chat_mode or "visitor_sales"), advisor_role="sales"
            )
            await self._chat_session_repository.append_message(session_id=sid, role="user", content=question)

        scope = self._compute_yuque_scope(owner, token_profile)
        generator = self._build_generator_by_selected_model(model or settings.llm_model)
        mcp_client = self._build_mcp_client(scope)
        orch = SalesDialogOrchestratorV3(
            mcp_client=mcp_client,
            generator=generator,
            profile_repo=self._chat_session_profile_repository,
            toc_nodes=self._guide_toc_nodes,
        )

        history = await self._chat_session_repository.list_recent_messages(session_id=sid, limit=20) if sid else []
        answer_parts: List[str] = []
        async for event in orch.answer_stream(question=question, session_id=sid, history=history):
            if event.get("event") == "token":
                try:
                    tok = str((event.get("data") or {}).get("token") or "")
                    if tok:
                        answer_parts.append(tok)
                except Exception:
                    pass
            if event.get("event") == "done":
                # 用累计 token 覆盖 answer，确保前端/存储一致
                data = event.get("data") or {}
                if isinstance(data, dict):
                    data["answer"] = "".join(answer_parts).strip() or str(data.get("answer") or "")
                if sid:
                    try:
                        ans = str(data.get("answer") or "")
                        if ans.strip():
                            await self._chat_session_repository.append_message(session_id=sid, role="assistant", content=ans)
                    except Exception:
                        pass
                yield {"event": "done", "data": data}
                continue
            yield event

    async def chat_v4_stream(
        self,
        question: str,
        *,
        model: Optional[str] = None,
        owner: Optional[str] = None,
        token_profile: Optional[str] = None,
        chat_mode: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """V4：目录状态机 + 目录内关联讲解（SSE）。"""
        await self._refresh_guide_doc_titles_if_stale()
        sid = (session_id or "").strip()
        if sid:
            await self._chat_session_repository.ensure_session(
                session_id=sid, chat_mode=(chat_mode or "visitor_sales"), advisor_role="sales"
            )
            await self._chat_session_repository.append_message(session_id=sid, role="user", content=question)

        scope = self._compute_yuque_scope(owner, token_profile)
        generator = self._build_generator_by_selected_model(model or settings.llm_model)
        mcp_client = self._build_mcp_client(scope)
        lead_outreach = V4LeadOutreach(
            lead_policy=self._lead_nudge_policy,
            lead_capture_repository=self._lead_capture_repository,
        )
        orch = SalesDialogOrchestratorV4(
            mcp_client=mcp_client,
            generator=generator,
            profile_repo=self._chat_session_profile_repository,
            toc_nodes=self._guide_toc_nodes,
            lead_outreach=lead_outreach,
        )

        history = await self._chat_session_repository.list_recent_messages(session_id=sid, limit=20) if sid else []
        answer_parts: List[str] = []
        async for event in orch.answer_stream(question=question, session_id=sid, history=history):
            if event.get("event") == "token":
                try:
                    tok = str((event.get("data") or {}).get("token") or "")
                    if tok:
                        answer_parts.append(tok)
                except Exception:
                    pass
            if event.get("event") == "done":
                data = event.get("data") or {}
                if isinstance(data, dict):
                    ans = "".join(answer_parts).strip() or str(data.get("answer") or "")
                    media_raw = data.get("media")
                    if media_raw and ans:
                        try:
                            bundle = (
                                ChatMediaBundle.model_validate(media_raw)
                                if isinstance(media_raw, dict)
                                else media_raw
                            )
                            ans = _strip_media_urls_from_text(ans, bundle)
                        except Exception:
                            pass
                    data["answer"] = ans
                if sid:
                    try:
                        ans = str(data.get("answer") or "")
                        if ans.strip():
                            await self._chat_session_repository.append_message(
                                session_id=sid, role="assistant", content=ans
                            )
                    except Exception:
                        pass
                yield {"event": "done", "data": data}
                continue
            yield event

    async def issue_v4_trial_credentials(self, *, session_id: str) -> TrialCredentialsResponse:
        sid = (session_id or "").strip()
        if not sid:
            return TrialCredentialsResponse(ok=False, message="缺少会话标识。")
        accounts = load_trial_accounts()
        if not accounts:
            return TrialCredentialsResponse(ok=False, message="暂未配置试用账号，请联系顾问。")
        profile = await self._chat_session_profile_repository.get_profile(session_id=sid)
        wants = False
        if profile and isinstance(profile.interests, dict):
            lead = profile.interests.get("_lead")
            if isinstance(lead, dict):
                wants = bool(lead.get("wants_trial"))
        if not wants:
            return TrialCredentialsResponse(
                ok=False,
                message="请先说明希望申请测试或试用，我再为您开通试用账号。",
            )
        picked = allocate_trial_account(sid, accounts)
        if not picked:
            return TrialCredentialsResponse(ok=False, message="试用账号分配失败，请稍后重试。")
        return TrialCredentialsResponse(
            ok=True,
            username=picked.username,
            password=picked.password,
            label=picked.label,
            message="以下为您的试用账号，请妥善保管。",
        )

    @staticmethod
    def _auto_skill_instruction_for_v2(question: str) -> str:
        q = (question or "").strip()
        if not q:
            return ""
        if any(k in q for k in ("怎么", "如何", "步骤", "流程", "上手")):
            return "请优先给出可执行步骤，并补充适用场景与注意事项。"
        if any(k in q for k in ("案例", "介绍", "了解", "看看", "是什么", "功能")):
            return "请先给出简洁总览，再给出2-3个可继续深入的问题方向。"
        return ""

    def runtime_mode(self) -> tuple[str, str]:
        if self._embedder is None:
            return "direct_yuque", "语雀直连模式"
        return "rag", "RAG 向量模式"

    async def _apply_visitor_sales_client_mask(
        self,
        *,
        question: str,
        session_id: Optional[str],
        response: ChatResponse,
    ) -> ChatResponse:
        contact = extract_contact(question)
        vt = detect_visitor_type(question)
        vt_s = vt if vt != "unknown" else None
        lead_saved = False
        sid = (session_id or "").strip()
        if contact and sid:
            lead_saved = await self._lead_capture_repository.try_insert_lead(
                session_id=sid,
                contact_type=contact.contact_type,
                contact_value=contact.value,
                visitor_type=vt_s,
            )
        dbg = dict(response.debug or {})
        dbg["visitor_sales"] = {
            "visitor_type": vt,
            "contact_detected": bool(contact),
            "lead_saved": lead_saved,
        }
        return response.model_copy(update={"sources": [], "debug": dbg})

    async def _run_one(
        self,
        question: str,
        *,
        model: Optional[str],
        owner: Optional[str],
        skill_id: Optional[str],
        generation_question: str,
        selected_yuque_docs: Optional[List[SelectedYuqueDocRef]] = None,
        token_profile: Optional[str] = None,
        chat_mode: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> ChatResponse:
        visitor = _is_visitor_sales(chat_mode)
        # 语雀作用域与向量检索作用域可能不一致：当 owner 非默认作用域时，强制走直连（embedder=None）。
        scope = self._compute_yuque_scope(owner, token_profile)
        embedder_for_retriever = self._embedder_for_profile(token_profile, scope)
        generator = self._build_generator_by_selected_model(model or settings.llm_model)

        yuque_loader = self._build_yuque_loader(scope, token_profile)
        mcp_client = self._build_mcp_client(scope)
        pipeline = RAGPipeline(
            retriever=Retriever(
                vector_store=self._vector_store,
                embedder=embedder_for_retriever,
                mcp_client=mcp_client,
                yuque_loader=yuque_loader,
                top_k=settings.top_k,
                score_threshold=settings.retrieval_score_threshold,
            ),
            generator=generator,
        )
        anchors = _doc_anchor_pairs(selected_yuque_docs)
        try:
            response = await pipeline.run(
                question,
                retrieval_question=question,
                generation_question=generation_question,
                skill_id=skill_id,
                doc_anchors=anchors,
                visitor_sales=visitor,
            )
        finally:
            await yuque_loader.close()

        if not settings.expose_source_urls:
            for source in response.sources:
                source.url = None
        logger.info(
            "chat_completed(dyn) sources=%d fallback_used=%s",
            len(response.sources),
            response.fallback_used,
        )
        await self._qa_log_repository.log_chat(question=question, response=response)
        if visitor:
            return await self._apply_visitor_sales_client_mask(
                question=question, session_id=session_id, response=response
            )
        return response

    async def _run_one_stream(
        self,
        question: str,
        *,
        model: Optional[str],
        owner: Optional[str],
        skill_id: Optional[str],
        generation_question: str,
        selected_yuque_docs: Optional[List[SelectedYuqueDocRef]] = None,
        token_profile: Optional[str] = None,
        chat_mode: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> AsyncIterator[dict[str, Any]]:
        visitor = _is_visitor_sales(chat_mode)
        scope = self._compute_yuque_scope(owner, token_profile)
        embedder_for_retriever = self._embedder_for_profile(token_profile, scope)
        generator = self._build_generator_by_selected_model(model or settings.llm_model)

        yuque_loader = self._build_yuque_loader(scope, token_profile)
        mcp_client = self._build_mcp_client(scope)
        pipeline = RAGPipeline(
            retriever=Retriever(
                vector_store=self._vector_store,
                embedder=embedder_for_retriever,
                mcp_client=mcp_client,
                yuque_loader=yuque_loader,
                top_k=settings.top_k,
                score_threshold=settings.retrieval_score_threshold,
            ),
            generator=generator,
        )
        anchors = _doc_anchor_pairs(selected_yuque_docs)

        completed = False
        try:
            dyn_mode, dyn_label = ("direct_yuque", "语雀直连模式") if embedder_for_retriever is None else ("rag", "RAG 向量模式")
            yield _sse_stage(
                "retrieving",
                _retrieving_stage_detail(runtime_label=dyn_label, skill_id=skill_id, scope=scope),
                mode=dyn_mode,
                skill_id=skill_id,
                scope=scope,
            )
            retrieval, debug = await pipeline.retrieve_context(
                question,
                retrieval_question=question,
                generation_question=generation_question,
                skill_id=skill_id,
                doc_anchors=anchors,
            )
            if (debug or {}).get("vision_images_used"):
                yield _sse_stage(
                    "vision",
                    _vision_stage_detail(debug),
                    vision_images_used=(debug or {}).get("vision_images_used"),
                    vision_model=(debug or {}).get("vision_model"),
                    scope=scope,
                )
            yield _sse_stage(
                "generating",
                _generating_stage_detail(debug),
                retrieval_mode=(debug or {}).get("retrieval_mode"),
            )
            answer_stream = pipeline.stream_answer_tokens(
                retrieval, generation_question=generation_question, visitor_sales=visitor
            )
            answer_parts: List[str] = []
            async for token in answer_stream:
                answer_parts.append(token)
                yield {"event": "token", "data": {"token": token}}

            answer = "".join(answer_parts) or "未生成回答。"
            response = ChatResponse(
                answer=answer,
                sources=retrieval.sources,
                fallback_used=retrieval.fallback_used,
                debug=debug,
            )
            if not settings.expose_source_urls:
                for source in response.sources:
                    source.url = None
            if response.debug is not None and skill_id:
                response.debug["skill_id"] = skill_id

            logger.info(
                "chat_completed(dyn) sources=%d fallback_used=%s",
                len(response.sources),
                response.fallback_used,
            )
            await self._qa_log_repository.log_chat(question=question, response=response)
            completed = True
            if visitor:
                response = await self._apply_visitor_sales_client_mask(
                    question=question, session_id=session_id, response=response
                )
            yield {"event": "done", "data": response.model_dump()}
        finally:
            if not completed:
                logger.info("chat_stream(dyn) ended_before_done owner=%r scope=%r", owner, scope)
            await yuque_loader.close()

    async def list_session_messages(self, *, session_id: str, limit: int) -> List[ChatMessageRow]:
        sid = (session_id or "").strip()
        if not sid:
            return []
        safe_limit = max(1, min(int(limit), 200))
        return await self._chat_session_repository.list_recent_messages(session_id=sid, limit=safe_limit)

    async def reset_session(self, *, session_id: str, chat_mode: str = "visitor_sales") -> None:
        sid = (session_id or "").strip()
        if not sid:
            return
        await self._chat_session_repository.reset_session(session_id=sid, chat_mode=chat_mode, advisor_role="sales")

    def guide_titles_state(self) -> dict[str, Any]:
        refresh_s = max(0, int(settings.chat_v15_guide_refresh_s))
        age_s: Optional[float] = None
        if self._guide_titles_refreshed_at > 0:
            age_s = max(0.0, time.monotonic() - self._guide_titles_refreshed_at)
        nodes = self._build_guide_toc_tree()
        max_level = 0
        for node in self._guide_toc_nodes:
            try:
                max_level = max(max_level, int(node.get("level") or 0))
            except Exception:
                continue
        return {
            "v15_enabled": bool(settings.chat_v15_enabled),
            "count": len(self._guide_doc_titles),
            "titles": list(self._guide_doc_titles),
            "total_nodes": len(self._guide_toc_nodes),
            "root_nodes": len(nodes),
            "max_level": max_level,
            "nodes": nodes,
            "refresh_interval_s": refresh_s,
            "refreshed_seconds_ago": age_s,
        }

    async def _history_block_for_session(self, session_id: str) -> str:
        """取最近 10 轮（=20 条消息）作为生成上下文；不用于向量检索。"""
        sid = (session_id or "").strip()
        if not sid:
            return ""
        msgs = await self._chat_session_repository.list_recent_messages(session_id=sid, limit=20)
        if not msgs:
            return ""
        lines: List[str] = []
        for m in msgs:
            role = "用户" if m.role == "user" else "助手"
            content = (m.content or "").strip()
            if not content:
                continue
            # 防止历史块无限膨胀：每条最多截断
            if len(content) > 800:
                content = content[:800] + "…"
            lines.append(f"{role}：{content}")
        if not lines:
            return ""
        body = "\n".join(lines)
        return (
            "【以下为本会话最近对话记录，仅用于你承接上下文与避免重复提问；"
            "不要逐字复述记录内容，也不要暴露内部分析标签。】\n"
            + body
        )

    @staticmethod
    def _with_history(generation_question: str, history_block: str) -> str:
        if not history_block.strip():
            return generation_question
        q = (generation_question or "").strip()
        return history_block + "\n\n本轮问题：\n" + q

    def _build_media_orchestrator(
        self,
        *,
        mcp_client: YuqueMCPClient,
        generator: Generator,
    ) -> MediaAnswerOrchestrator:
        return MediaAnswerOrchestrator(
            mcp_client=mcp_client,
            generator=generator,
            lead_policy=self._lead_nudge_policy,
            lead_capture_repository=self._lead_capture_repository,
            chat_session_repository=self._chat_session_repository,
            max_images=settings.chat_v15_max_images,
            max_videos=settings.chat_v15_max_videos,
            max_docs=settings.chat_v15_max_docs,
            prefetched_titles=self._guide_doc_titles,
            prefetched_toc_nodes=self._guide_toc_nodes,
            image_rerank_mode=("rule" if settings.chat_v15_image_rerank_mode == "rule" else "text_rerank"),
        )

    async def _warmup_guide_doc_titles(self) -> None:
        """启动时预加载语雀目录标题，用于 V1.5 的提问引导。"""
        titles: List[str] = []
        toc_nodes_data: List[Dict[str, Any]] = []
        scope = (settings.yuque_scope or "").strip().strip("/")
        if not scope:
            self._guide_doc_titles = []
            self._guide_toc_nodes = []
            return
        try:
            toc_nodes = await self._yuque_loader.get_book_toc(book=scope)
            for node in toc_nodes:
                title = (getattr(node, "title", "") or "").strip()
                if not title:
                    continue
                raw_level = getattr(node, "level", 1)
                try:
                    level = max(1, int(raw_level or 1))
                except Exception:
                    level = 1
                toc_nodes_data.append(
                    {
                        "uuid": str(getattr(node, "uuid", "") or ""),
                        "title": title,
                        "level": level,
                        "parent_uuid": str(getattr(node, "parent_uuid", "") or ""),
                        "node_type": str(getattr(node, "type", "") or ""),
                        "url": (getattr(node, "url", "") or "").strip() or None,
                        "doc_id": getattr(node, "doc_id", None),
                    }
                )
                titles.append(title)
                if len(titles) >= 80:
                    break
        except Exception:
            titles = []
            toc_nodes_data = []
        if not titles:
            try:
                docs = await self._yuque_loader.list_docs(book=scope, offset=0, limit=80)
                titles = [(getattr(d, "title", "") or "").strip() for d in docs]
                toc_nodes_data = []
                for idx, d in enumerate(docs):
                    title = (getattr(d, "title", "") or "").strip()
                    if not title:
                        continue
                    doc_id = getattr(d, "id", None)
                    node_uuid = f"doc-{doc_id or idx}"
                    toc_nodes_data.append(
                        {
                            "uuid": node_uuid,
                            "title": title,
                            "level": 1,
                            "parent_uuid": "",
                            "node_type": "doc",
                            "url": (getattr(d, "url", "") or "").strip() or None,
                            "doc_id": doc_id,
                        }
                    )
            except Exception:
                titles = []
                toc_nodes_data = []
        seen: set[str] = set()
        dedup: List[str] = []
        for t in titles:
            if not t or t in seen:
                continue
            seen.add(t)
            dedup.append(t)
        self._guide_doc_titles = dedup[:80]
        self._guide_toc_nodes = toc_nodes_data[:200]
        self._guide_titles_refreshed_at = time.monotonic()

    def _build_guide_toc_tree(self) -> List[Dict[str, Any]]:
        if not self._guide_toc_nodes:
            return []
        nodes: List[Dict[str, Any]] = []
        by_uuid: Dict[str, Dict[str, Any]] = {}
        for item in self._guide_toc_nodes:
            node = {
                "uuid": str(item.get("uuid") or ""),
                "title": str(item.get("title") or ""),
                "level": int(item.get("level") or 1),
                "parent_uuid": str(item.get("parent_uuid") or ""),
                "node_type": str(item.get("node_type") or ""),
                "url": item.get("url"),
                "doc_id": item.get("doc_id"),
                "children": [],
            }
            nodes.append(node)
            if node["uuid"] and node["uuid"] not in by_uuid:
                by_uuid[node["uuid"]] = node
        roots: List[Dict[str, Any]] = []
        for node in nodes:
            parent_uuid = node["parent_uuid"]
            if parent_uuid and parent_uuid in by_uuid and parent_uuid != node["uuid"]:
                by_uuid[parent_uuid]["children"].append(node)
                continue
            roots.append(node)
        return roots

    async def _refresh_guide_doc_titles_if_stale(self, *, force: bool = False) -> None:
        refresh_s = max(0, int(settings.chat_v15_guide_refresh_s))
        now = time.monotonic()
        if not force and refresh_s > 0 and self._guide_doc_titles:
            if (now - self._guide_titles_refreshed_at) < float(refresh_s):
                return
        async with self._guide_titles_refresh_lock:
            now = time.monotonic()
            if not force and refresh_s > 0 and self._guide_doc_titles:
                if (now - self._guide_titles_refreshed_at) < float(refresh_s):
                    return
            await self._warmup_guide_doc_titles()

    def _start_guide_titles_refresh_loop(self) -> None:
        refresh_s = max(0, int(settings.chat_v15_guide_refresh_s))
        if refresh_s <= 0:
            logger.info("guide_titles_auto_refresh_disabled interval_s=%s", refresh_s)
            return
        if self._guide_titles_refresh_task and not self._guide_titles_refresh_task.done():
            return
        self._guide_titles_refresh_task = asyncio.create_task(
            self._guide_titles_refresh_loop(refresh_s),
            name="guide-titles-refresh",
        )

    async def _stop_guide_titles_refresh_loop(self) -> None:
        task = self._guide_titles_refresh_task
        self._guide_titles_refresh_task = None
        if not task:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def _guide_titles_refresh_loop(self, refresh_s: int) -> None:
        while True:
            await asyncio.sleep(refresh_s)
            try:
                await self._refresh_guide_doc_titles_if_stale(force=True)
            except Exception as exc:
                logger.warning("guide_titles_auto_refresh_failed err=%s", exc)

    def _compute_yuque_scope(self, owner: Optional[str], token_profile: Optional[str] = None) -> str:
        default_scope = default_yuque_scope_for_profile(token_profile).strip().strip("/")
        if not owner:
            return default_scope
        owner = owner.strip().strip("/")
        if not default_scope or "/" not in default_scope:
            return owner
        _, repo = default_scope.split("/", 1)
        return f"{owner}/{repo}"

    def _embedder_for_profile(self, token_profile: Optional[str], scope: str) -> Optional[Embedder]:
        """副账号 Token 与主索引向量空间不一致，一律走语雀直连，避免误召回。"""
        if normalize_yuque_token_profile(token_profile) == "secondary":
            return None
        return self._embedder if scope == settings.yuque_scope else None

    def _build_yuque_loader(self, scope: str, token_profile: Optional[str] = None) -> YuqueLoader:
        return YuqueLoader(
            token=yuque_token_for_profile(token_profile),
            base_url=settings.yuque_base_url,
            timeout_s=settings.yuque_timeout_s,
            scope=scope,
        )

    def _build_mcp_client(self, scope: str) -> YuqueMCPClient:
        return YuqueMCPClient(
            command=settings.mcp_server_command,
            args=settings.mcp_server_args,
            repo_id=scope,
            search_tool=settings.mcp_search_tool,
            get_doc_tool=settings.mcp_get_doc_tool,
        )

    def _build_generator_by_selected_model(self, model: str) -> Generator:
        normalized = (model or "").strip()
        if not normalized:
            raise GeneratorConfigError("模型不能为空。")

        # 前端目前只会传入 deepseek-* 与 gpt-*；用前缀判断应走哪个供应商 key。
        is_openai = normalized.lower().startswith("gpt-")
        if is_openai:
            if not settings.openai_api_key:
                raise GeneratorConfigError(f"缺少 OPENAI_API_KEY，无法使用模型 {normalized}。")
            return OpenAIGenerator(model=normalized, api_key=settings.openai_api_key, base_url=settings.openai_base_url)

        if not settings.deepseek_api_key:
            raise GeneratorConfigError(f"缺少 DEEPSEEK_API_KEY，无法使用模型 {normalized}。")
        return DeepSeekGenerator(
            model=normalized,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )

    def mcp_capabilities(self) -> dict[str, Any]:
        """与 YuqueMCPClient.read_tools / call_raw 白名单对齐：只读工具为 integrated，写操作未接入为 available。"""
        wired = set(self._mcp_client.read_tools)

        def _is_wired(display_name: str) -> bool:
            if display_name in wired:
                return True
            # 展示名固定为 yuque_*，.env 里可能配置为 search / get_doc
            if display_name == "yuque_get_doc":
                return (settings.mcp_get_doc_tool or "yuque_get_doc") in wired
            if display_name == "yuque_search":
                return (settings.mcp_search_tool or "yuque_search") in wired
            return False

        tool_items = [
            ("yuque_get_user", "user", "读取当前语雀用户信息"),
            ("yuque_list_books", "book", "列出可访问知识库"),
            ("yuque_get_book", "book", "读取知识库详情"),
            ("yuque_create_book", "book", "创建知识库"),
            ("yuque_update_book", "book", "更新知识库"),
            ("yuque_list_docs", "doc", "列出知识库文档"),
            ("yuque_get_doc", "doc", "读取文档正文"),
            ("yuque_create_doc", "doc", "创建文档"),
            ("yuque_update_doc", "doc", "更新文档"),
            ("yuque_get_toc", "toc", "读取目录结构"),
            ("yuque_update_toc", "toc", "更新目录结构"),
            ("yuque_search", "search", "全文搜索文档/知识库"),
            ("yuque_list_notes", "note", "列出小记"),
            ("yuque_get_note", "note", "读取小记"),
            ("yuque_create_note", "note", "创建小记"),
            ("yuque_update_note", "note", "更新小记"),
        ]
        tools = [
            {
                "name": name,
                "category": category,
                "status": "integrated" if _is_wired(name) else "available",
                "description": description,
            }
            for name, category, description in tool_items
        ]
        return {
            "enabled": self._mcp_client.enabled,
            "repo_scope": settings.yuque_scope,
            "secondary_token_configured": secondary_yuque_configured(),
            "repo_scope_secondary": default_yuque_scope_for_profile("secondary")
            if secondary_yuque_configured()
            else "",
            "yuque_scope_secondary_explicit": bool((settings.yuque_scope_secondary or "").strip()),
            "tools": tools,
        }

    async def resolve_yuque_token_logins(self) -> tuple[str, str]:
        """各 Token 请求语雀 /user，返回 (主 login, 副 login)；失败则为空串。"""
        timeout = min(float(settings.yuque_timeout_s), 10.0)

        async def _one(token: str) -> str:
            tok = (token or "").strip()
            if not tok:
                return ""
            ld = YuqueLoader(
                token=tok,
                base_url=settings.yuque_base_url,
                timeout_s=timeout,
                scope="",
            )
            try:
                return await ld.fetch_self_login()
            except YuqueLoaderError:
                return ""
            finally:
                await ld.close()

        pri = await _one(yuque_token_for_profile("primary"))
        sec = ""
        if secondary_yuque_configured():
            sec = await _one(yuque_token_for_profile("secondary"))
        return pri, sec

    async def rebuild_index(self, *, bootstrap_query: str) -> tuple[int, int]:
        if self._embedder is None:
            return 0, 0
        documents = await self._yuque_loader.fetch_documents_for_bootstrap(query=bootstrap_query)
        chunks = self._chunk_documents(documents)
        embeddings = await self._embedder.embed_texts([chunk.text for chunk in chunks])
        stored_chunks = [
            StoredChunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                title=chunk.title,
                url=chunk.url,
                text=chunk.text,
                order=chunk.order,
            )
            for chunk in chunks
        ]
        self._vector_store.rebuild(chunks=stored_chunks, embeddings=embeddings)
        await self._document_repository.replace_documents(chunks)
        return len(documents), len(chunks)

    def _chunk_documents(self, documents: List[YuqueDocument]) -> List[TextChunk]:
        chunks: List[TextChunk] = []
        for document in documents:
            chunks.extend(
                self._splitter.split_document(
                    doc_id=document.doc_id,
                    title=document.title,
                    url=document.url,
                    text=document.body,
                )
            )
        return chunks

    def _build_embedder(self) -> Embedder | None:
        if not settings.embedding_api_key:
            return None
        if settings.embedding_provider.lower() == "bge-small":
            return BGESmallEmbedder()
        return OpenAIEmbedder(
            model=settings.embedding_model,
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url,
        )

    def _build_generator(self) -> Generator | None:
        if not settings.llm_api_key:
            return None
        if settings.llm_provider.lower() == "openai":
            return OpenAIGenerator(
                model=settings.llm_model,
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
            )
        return DeepSeekGenerator(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )

