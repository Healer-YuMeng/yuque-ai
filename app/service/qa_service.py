from __future__ import annotations

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
from app.conversation.visitor_prompt import build_visitor_generation_question
from app.conversation.visitor_profile import detect_visitor_type
from app.db.repositories import ChatMessageRow, ChatSessionRepository, DocumentRepository, LeadCaptureRepository, QALogRepository
from app.rag.embedder import BGESmallEmbedder, Embedder, OpenAIEmbedder
from app.rag.generator import DeepSeekGenerator, Generator, GeneratorConfigError, OpenAIGenerator
from app.rag.skill_router import route_skill
from app.rag.pipeline import RAGPipeline
from app.rag.retriever import Retriever
from app.schemas.chat import ChatResponse, SelectedYuqueDocRef
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

    async def startup(self) -> None:
        await self._document_repository.init_db()
        # 最小实现：启动时做一次过期清理，避免数据无限增长（默认 7 天）。
        await self._chat_session_repository.prune_older_than_days(retention_days=7)

    async def shutdown(self) -> None:
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

