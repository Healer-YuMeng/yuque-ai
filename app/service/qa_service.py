from __future__ import annotations

import asyncio
import contextlib
import re
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
from app.conversation.chat_display import display_name_for_chat, normalize_display_name
from app.conversation.contact_extractor import extract_contact
from app.conversation.friend_persona_v5 import all_friend_v5_scenes
from app.conversation.lead_nudge_policy import LeadNudgePolicy
from app.conversation.trial_account_pool import allocate_trial_account, load_trial_accounts
from app.conversation.user_info_extractor import UserInfoStructuredExtractor
from app.conversation.user_info_cleaner import (
    extract_email_text_candidate,
    normalize_display_name_candidate,
    normalize_email_candidate,
    normalize_organization_candidate,
)
from app.conversation.v4_lead_outreach import V4LeadOutreach
from app.conversation.visitor_prompt import build_visitor_generation_question
from app.conversation.visitor_profile import detect_visitor_type
from app.db.repositories import (
    AdminVideoAssetRepository,
    ChatMessageRow,
    ChatSessionRepository,
    DocumentRepository,
    LeadCaptureRepository,
    QALogRepository,
)
from app.db.profile_repository import ChatSessionProfileRepository
from app.rag.embedder import BGESmallEmbedder, Embedder, OpenAIEmbedder
from app.rag.friend_v5_generator import FriendV5Generator
from app.rag.generator import DeepSeekGenerator, Generator, GeneratorConfigError, OpenAIGenerator
from app.rag.skill_router import route_skill
from app.rag.pipeline import RAGPipeline
from app.rag.retriever import Retriever
from app.schemas.chat import (
    ChatMediaBundle,
    ChatResponse,
    ChatV2Response,
    ChatV4Response,
    SelectedYuqueDocRef,
    TrialCredentialsResponse,
    VisitorProfileResponse,
)
from app.service.media_answer_orchestrator import MediaAnswerOrchestrator
from app.service.friend_dialog_orchestrator_v5 import FriendDialogOrchestratorV5
from app.service.friend_v5_scene_query_rewriter import FriendV5SceneQueryRewriter
from app.service.friend_v5_yuque_deep_reader import FriendV5YuqueDeepReader
from app.service.sales_dialog_orchestrator_v3 import SalesDialogOrchestratorV3
from app.service.sales_dialog_orchestrator_v4 import SalesDialogOrchestratorV4, _strip_media_urls_from_text
from app.storage.vector_store import VectorStore
from app.core.logger import get_logger

logger = get_logger(__name__)

_EMAIL_RE = re.compile(r"^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$", re.IGNORECASE)


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


def _is_v4_memory_only_question(question: str) -> bool:
    q = (question or "").strip()
    if not q:
        return False
    memory_hits = (
        "你知道我的电话",
        "你知道我电话",
        "你记得我的电话",
        "你知道我的联系方式",
        "你记得我的联系方式",
        "我的联系方式已经给你",
        "联系方式已经给你",
        "电话已经给你",
        "我已经给过",
        "我刚才给过",
        "你知道我的姓名",
        "你知道我姓名",
        "你知道我的名字",
        "你记得我的名字",
        "你知道我的单位",
        "你记得我的单位",
    )
    return any(hit in q for hit in memory_hits)


def _visitor_profile_parts(profile: Any) -> dict[str, str]:
    if not profile:
        return {"name": "", "org_name": "", "contact": "", "email": "", "interested_product": "", "concern": "", "module_scope": ""}
    interests = profile.interests if isinstance(profile.interests, dict) else {}
    lead = interests.get("_lead") if isinstance(interests.get("_lead"), dict) else {}
    session_meta = interests.get("_session") if isinstance(interests.get("_session"), dict) else {}
    org_name = str(getattr(profile, "org_name", "") or lead.get("org_name") or "")
    fallback_name = normalize_display_name(str(lead.get("name") or ""), org=org_name)
    return {
        "name": display_name_for_chat(profile) or fallback_name,
        "org_name": org_name,
        "contact": str(lead.get("contact_value") or ""),
        "email": str(lead.get("email") or ""),
        "interested_product": str(lead.get("interested_product") or ""),
        "concern": str(lead.get("concern") or ""),
        "module_scope": str(session_meta.get("module_scope") or ""),
    }


def _trial_apply_info_transcript(*, name: str, org_name: str, contact: str, email: str) -> str:
    return "\n".join(
        [
            f"姓名：{(name or '').strip()}",
            f"单位：{(org_name or '').strip()}",
            f"联系方式：{(contact or '').strip()}",
            f"邮箱：{(email or '').strip()}",
        ]
    ).strip()


def _build_v4_memory_answer(profile: Any) -> str:
    parts = _visitor_profile_parts(profile)
    name = parts["name"]
    contact = parts["contact"]
    org = parts["org_name"]
    product = parts["interested_product"]
    prefix = f"{name}，" if name else ""
    known: List[str] = []
    if name:
        known.append(f"姓名/称呼是{name}")
    if contact:
        known.append(f"联系方式是{contact}")
    if org:
        known.append(f"单位是{org}")
    if product:
        known.append(f"关注方向是{product}")
    if not known:
        return "我这边还没有记录到您的姓名、单位或联系方式。您可以先告诉我其中一项，我会继续记住后面的沟通。"
    return f"{prefix}我记得，您之前提供的信息包括：" + "，".join(known) + "。后续我会结合这些信息继续沟通，不会重复向您索要已提供的内容。"


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
        admin_video_asset_repository: Optional[AdminVideoAssetRepository] = None,
        admin_scene_intro_repository: Optional[Any] = None,
    ) -> None:
        self._yuque_loader = yuque_loader
        self._vector_store = vector_store
        self._document_repository = document_repository
        self._qa_log_repository = qa_log_repository
        self._lead_capture_repository = lead_capture_repository
        self._chat_session_repository = chat_session_repository
        self._admin_video_asset_repository = admin_video_asset_repository
        self._admin_scene_intro_repository = admin_scene_intro_repository
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
        self._user_info_extractor = UserInfoStructuredExtractor()
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
        selected_yuque_docs: Optional[List[SelectedYuqueDocRef]] = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """V4：目录状态机 + 目录内关联讲解（SSE）。"""
        sid = (session_id or "").strip()
        history: List[ChatMessageRow] = []
        if sid:
            await self._chat_session_repository.ensure_session(
                session_id=sid, chat_mode=(chat_mode or "visitor_sales"), advisor_role="sales"
            )
            history = await self._chat_session_repository.list_recent_messages(session_id=sid, limit=20)
            await self._chat_session_repository.append_message(session_id=sid, role="user", content=question)
            if _is_v4_memory_only_question(question):
                profile = await self._chat_session_profile_repository.get_profile(session_id=sid)
                answer = _build_v4_memory_answer(profile)
                yield _sse_stage("memory", "正在根据本会话已收集的信息回答…", retrieval_skipped=True)
                for token in answer:
                    yield {"event": "token", "data": {"token": token}}
                data = ChatV4Response(
                    answer=answer,
                    sources=[],
                    fallback_used=False,
                    debug={"pipeline": "v4_memory", "retrieval_skipped": True},
                    lead_nudge_triggered=False,
                    trial_apply_available=False,
                ).model_dump()
                await self._chat_session_repository.append_message(session_id=sid, role="assistant", content=answer)
                yield {"event": "done", "data": data}
                return

        await self._refresh_guide_doc_titles_if_stale()

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

        answer_parts: List[str] = []
        selected_doc_ids = [int(x.doc_id) for x in (selected_yuque_docs or []) if int(x.doc_id) >= 1]
        async for event in orch.answer_stream(
            question=question,
            session_id=sid,
            history=history,
            selected_doc_ids=selected_doc_ids,
        ):
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

    async def chat_v5_stream(
        self,
        question: str,
        *,
        model: Optional[str] = None,
        owner: Optional[str] = None,
        token_profile: Optional[str] = None,
        chat_mode: Optional[str] = None,
        session_id: Optional[str] = None,
        scene: str,
        trigger_type: str,
    ) -> AsyncIterator[dict[str, Any]]:
        """V5：知心朋友小为 + 联网搜索来源（SSE）。"""
        if not settings.chat_v5_enabled:
            yield {"event": "error", "data": {"message": "V5 链路未开启，请先设置 CHAT_V5_ENABLED=true。"}}
            return
        sid = (session_id or "").strip()
        if not sid.startswith("sess_v5_"):
            yield {"event": "error", "data": {"message": "V5 session_id 必须以 sess_v5_ 开头。"}}
            return
        if scene not in all_friend_v5_scenes():
            yield {"event": "error", "data": {"message": "V5 scene 不在允许范围内。"}}
            return
        if trigger_type not in {"scene", "tag", "manual"}:
            yield {"event": "error", "data": {"message": "V5 trigger_type 必须是 scene、tag 或 manual。"}}
            return

        await self._chat_session_repository.ensure_session(
            session_id=sid,
            chat_mode=(chat_mode or "friend_v5"),
            advisor_role="friend",
        )
        history = await self._chat_session_repository.list_recent_messages(session_id=sid, limit=20)
        await self._chat_session_repository.append_message(session_id=sid, role="user", content=question)

        selected_model = (model or settings.chat_v5_model or "qwen3.7-plus").strip()
        if not settings.is_dashscope_model(selected_model):
            raise GeneratorConfigError(f"V5 当前仅支持 DashScope 千问模型，收到 {selected_model}。")
        api_key, _ = settings.resolve_model_endpoint(selected_model)
        generator = FriendV5Generator(
            api_key=api_key,
            model=selected_model,
            generation_url=settings.chat_v5_generation_url,
            search_strategy=settings.chat_v5_search_strategy,
            max_tokens=settings.chat_v5_max_tokens,
            require_web_sources=(settings.chat_v5_require_web_sources and settings.chat_v5_web_search_enabled),
        )
        scene_query_rewriter = FriendV5SceneQueryRewriter(
            api_key=api_key,
            model=selected_model,
            generation_url=settings.chat_v5_generation_url,
        )
        scope = self._compute_yuque_scope(owner, token_profile)
        mcp_client = self._build_mcp_client(scope)
        yuque_loader = self._build_yuque_loader(scope, token_profile)
        yuque_search: Any = mcp_client if mcp_client.enabled else yuque_loader
        yuque_deep_reader = (
            FriendV5YuqueDeepReader(
                mcp_client=mcp_client,
                yuque_loader=yuque_loader,
                scope=scope,
                max_images=settings.chat_v5_max_images,
                max_videos=settings.chat_v5_max_videos,
            )
            if settings.chat_v5_yuque_deep_read_enabled
            else None
        )
        orch = FriendDialogOrchestratorV5(
            generator=generator,
            profile_repo=self._chat_session_profile_repository,
            yuque_search=yuque_search,
            scene_query_rewriter=scene_query_rewriter,
            yuque_deep_reader=yuque_deep_reader,
            admin_video_repository=self._admin_video_asset_repository,
            admin_scene_intro_repository=self._admin_scene_intro_repository,
            toc_nodes=self._guide_toc_nodes,
            yuque_url_limit=settings.chat_v5_yuque_url_limit,
            require_web_sources=settings.chat_v5_require_web_sources,
        )

        try:
            async for event in orch.answer_stream(
                question=question,
                session_id=sid,
                scene=scene,
                trigger_type=trigger_type,
                history=history,
            ):
                if event.get("event") == "done":
                    data = event.get("data") or {}
                    if isinstance(data, dict):
                        answer = str(data.get("answer") or "")
                        if answer.strip():
                            await self._chat_session_repository.append_message(
                                session_id=sid,
                                role="assistant",
                                content=answer,
                            )
                        lead_saved = await self._persist_v5_chat_lead_for_admin(
                            session_id=sid,
                            question=question,
                            scene=scene,
                        )
                        debug = dict(data.get("debug") or {})
                        debug["v5_lead_admin"] = {"lead_saved": lead_saved}
                        data["debug"] = debug
                    yield event
                    continue
                yield event
        finally:
            if not mcp_client.enabled:
                with contextlib.suppress(Exception):
                    await yuque_loader.close()

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
        if profile and isinstance(profile.interests, dict):
            interests = dict(profile.interests)
            session_meta = dict(interests.get("_session") or {})
            session_meta["module_scope"] = "使用指南"
            session_meta["trial_account_issued"] = True
            interests["_session"] = session_meta
            await self._chat_session_profile_repository.upsert_profile(session_id=sid, interests=interests)
        return TrialCredentialsResponse(
            ok=True,
            username=picked.username,
            password=picked.password,
            label=picked.label,
            message="以下为您的试用账号，请妥善保管。",
        )

    async def apply_visitor_trial_account(
        self,
        *,
        session_id: str,
        name: str,
        org_name: str,
        contact: str,
        email: str = "",
        interested_product: str = "",
        concern: str = "",
    ) -> TrialCredentialsResponse:
        sid = (session_id or "").strip()
        info_transcript = _trial_apply_info_transcript(
            name=name,
            org_name=org_name,
            contact=contact,
            email=email,
        )
        extractor = getattr(self, "_user_info_extractor", None) or UserInfoStructuredExtractor()
        structured_info = await extractor.extract(info_transcript)
        display_name = structured_info.display_name or normalize_display_name_candidate(name) or ""
        org = structured_info.org_name or normalize_organization_candidate(org_name) or ""
        contact_text = (contact or "").strip()
        if structured_info.contact:
            contact_text = structured_info.contact
        raw_email = structured_info.email or (email or "").strip()
        email_text = normalize_email_candidate(raw_email) or ""
        product = (interested_product or "").strip()
        concern_text = (concern or "").strip()
        if not sid:
            return TrialCredentialsResponse(ok=False, message="缺少会话标识，请刷新后重试。")
        if not (display_name and org and (contact_text or email_text)):
            return TrialCredentialsResponse(ok=False, message="请完整填写姓名、单位，并至少填写联系方式或邮箱。")
        contact_hit = extract_contact(contact_text) if contact_text else None
        if contact_text and not contact_hit:
            return TrialCredentialsResponse(ok=False, message="联系方式校验失败，请填写手机号或微信号。")
        if raw_email and not email_text:
            return TrialCredentialsResponse(ok=False, message="邮箱格式校验失败，请检查后重试。")

        lead_contact_type = contact_hit.contact_type if contact_hit else ""
        lead_contact_value = contact_hit.value if contact_hit else ""
        capture_contact_type = lead_contact_type or ("email" if email_text else "")
        capture_contact_value = lead_contact_value or email_text
        await self._lead_capture_repository.try_insert_lead(
            session_id=sid,
            contact_type=capture_contact_type,
            contact_value=capture_contact_value,
            visitor_type=None,
        )
        profile = await self._chat_session_profile_repository.get_profile(session_id=sid)
        interests = dict(profile.interests) if profile and isinstance(profile.interests, dict) else {}
        if not product:
            lead_existing = interests.get("_lead") if isinstance(interests.get("_lead"), dict) else {}
            session_existing = interests.get("_session") if isinstance(interests.get("_session"), dict) else {}
            product = str(
                lead_existing.get("interested_product")
                or session_existing.get("module_scope")
                or ""
            ).strip()
        lead = dict(interests.get("_lead") or {})
        lead.update(
            {
                "wants_trial": True,
                "name": display_name,
                "org_name": org,
                "contact_type": lead_contact_type,
                "contact_value": lead_contact_value,
                "email": email_text,
                "interested_product": product,
                "concern": concern_text,
            }
        )
        interests["_lead"] = lead
        session_meta = dict(interests.get("_session") or {})
        session_meta["module_scope"] = "使用指南"
        session_meta["trial_apply_submitted"] = True
        interests["_session"] = session_meta
        admin_meta = dict(interests.get("_admin") or {})
        admin_meta.setdefault("follow_up_status", "待跟进")
        admin_meta.setdefault("test_account_status", "待发放")
        interests["_admin"] = admin_meta
        await self._chat_session_profile_repository.upsert_profile(
            session_id=sid,
            display_name=display_name,
            org_name=org,
            interests=interests,
        )

        return TrialCredentialsResponse(
            ok=True,
            message="提交成功，我们会尽快与您联系。",
        )

    async def visitor_profile_summary(self, *, session_id: str) -> VisitorProfileResponse:
        sid = (session_id or "").strip()
        if not sid:
            return VisitorProfileResponse(ok=False)
        profile = await self._chat_session_profile_repository.get_profile(session_id=sid)
        parts = _visitor_profile_parts(profile)
        interests = profile.interests if profile and isinstance(profile.interests, dict) else {}
        session_meta = interests.get("_session") if isinstance(interests.get("_session"), dict) else {}
        return VisitorProfileResponse(
            ok=True,
            name=parts["name"],
            org_name=parts["org_name"],
            contact=parts["contact"],
            email=parts["email"],
            interested_product=parts["interested_product"],
            concern=parts.get("concern", ""),
            module_scope=parts["module_scope"],
            trial_account_issued=bool(session_meta.get("trial_account_issued")),
        )

    async def _persist_v5_chat_lead_for_admin(
        self,
        *,
        session_id: str,
        question: str,
        scene: str,
    ) -> bool:
        """把 V5 普通对话中留下的联系方式同步到后台客户管理。"""
        sid = (session_id or "").strip()
        if not sid:
            return False
        profile = await self._chat_session_profile_repository.get_profile(session_id=sid)
        interests = dict(profile.interests) if profile and isinstance(profile.interests, dict) else {}
        lead = dict(interests.get("_lead") or {})
        contact_hit = extract_contact(question)
        email_text = extract_email_text_candidate(question) or str(lead.get("email") or "").strip()
        contact_type = contact_hit.contact_type if contact_hit else str(lead.get("contact_type") or "").strip()
        contact_value = contact_hit.value if contact_hit else str(lead.get("contact_value") or "").strip()
        if (not contact_type or not contact_value) and email_text:
            contact_type = "email"
            contact_value = email_text
        if not contact_type or not contact_value:
            return False

        visitor_type = str(getattr(profile, "visitor_type", "") or "").strip()
        if not visitor_type:
            detected = detect_visitor_type(question)
            visitor_type = detected if detected != "unknown" else ""
        saved = await self._lead_capture_repository.try_insert_lead(
            session_id=sid,
            contact_type=contact_type,
            contact_value=contact_value,
            visitor_type=visitor_type or None,
        )

        display_name = str(getattr(profile, "display_name", "") or lead.get("name") or "").strip()
        org_name = str(getattr(profile, "org_name", "") or lead.get("org_name") or "").strip()
        lead.update(
            {
                "contact_type": contact_type,
                "contact_value": contact_value,
                "interested_product": str(lead.get("interested_product") or scene or "").strip(),
            }
        )
        if email_text:
            lead["email"] = email_text
        if display_name:
            lead["name"] = display_name
        if org_name:
            lead["org_name"] = org_name
        interests["_lead"] = lead
        admin_meta = dict(interests.get("_admin") or {})
        admin_meta.setdefault("follow_up_status", "待跟进")
        admin_meta.setdefault("test_account_status", "待发放")
        interests["_admin"] = admin_meta
        await self._chat_session_profile_repository.upsert_profile(
            session_id=sid,
            display_name=display_name or None,
            org_name=org_name or None,
            interests=interests,
        )
        return saved

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

    async def reset_session(self, *, session_id: str, chat_mode: str = "visitor_sales") -> None:
        sid = (session_id or "").strip()
        if not sid:
            return
        advisor_role = "friend" if chat_mode == "friend_v5" else "sales"
        await self._chat_session_repository.reset_session(
            session_id=sid,
            chat_mode=chat_mode,
            advisor_role=advisor_role,
        )

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

    async def refresh_guide_titles(self, *, force: bool = False) -> None:
        await self._refresh_guide_doc_titles_if_stale(force=force)

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
        self._guide_toc_nodes = toc_nodes_data
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

        api_key, base_url = settings.resolve_model_endpoint(normalized)
        if not api_key:
            if settings.is_openai_model(normalized):
                raise GeneratorConfigError(f"缺少 OPENAI_API_KEY，无法使用模型 {normalized}。")
            if settings.is_deepseek_model(normalized):
                raise GeneratorConfigError(f"缺少 DEEPSEEK_API_KEY，无法使用模型 {normalized}。")
            if settings.is_dashscope_model(normalized):
                raise GeneratorConfigError(f"缺少 DASHSCOPE_API_KEY，无法使用模型 {normalized}。")
            raise GeneratorConfigError(f"缺少 LLM_API_KEY，无法使用模型 {normalized}。")

        if settings.is_deepseek_model(normalized):
            logger.info("构建生成器 model=%s endpoint=%s", normalized, base_url or "default")
            return DeepSeekGenerator(model=normalized, api_key=api_key, base_url=base_url)
        logger.info("构建生成器 model=%s endpoint=%s", normalized, base_url or "default")
        return OpenAIGenerator(model=normalized, api_key=api_key, base_url=base_url)

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

    def chat_v5_capabilities(self) -> dict[str, Any]:
        return {
            "enabled": bool(settings.chat_v5_enabled),
            "model": settings.chat_v5_model,
            "require_web_sources": bool(settings.chat_v5_require_web_sources),
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
        await self._document_repository.replace_documents(chunks, embeddings=embeddings)
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
        api_key, base_url = settings.resolve_model_endpoint(settings.llm_model)
        if not api_key:
            return None
        if settings.is_deepseek_model(settings.llm_model):
            return DeepSeekGenerator(
                model=settings.llm_model,
                api_key=api_key,
                base_url=base_url,
            )
        return OpenAIGenerator(
            model=settings.llm_model,
            api_key=api_key,
            base_url=base_url,
        )
