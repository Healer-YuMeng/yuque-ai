from __future__ import annotations

from typing import Any, AsyncIterator, List, Optional

from app.core.config import settings
from app.data.mcp_client import YuqueMCPClient
from app.data.splitter import RecursiveTextSplitter, TextChunk
from app.data.yuque_loader import YuqueDocument, YuqueLoader
from app.db.repositories import DocumentRepository, QALogRepository
from app.rag.embedder import BGESmallEmbedder, Embedder, OpenAIEmbedder
from app.rag.generator import DeepSeekGenerator, Generator, GeneratorConfigError, OpenAIGenerator
from app.rag.skill_router import route_skill
from app.rag.pipeline import RAGPipeline
from app.rag.retriever import Retriever
from app.schemas.chat import ChatResponse
from app.storage.vector_store import StoredChunk, VectorStore
from app.core.logger import get_logger

logger = get_logger(__name__)


class QAService:
    def __init__(
        self,
        *,
        yuque_loader: YuqueLoader,
        vector_store: VectorStore,
        document_repository: DocumentRepository,
        qa_log_repository: QALogRepository,
    ) -> None:
        self._yuque_loader = yuque_loader
        self._vector_store = vector_store
        self._document_repository = document_repository
        self._qa_log_repository = qa_log_repository
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

    async def shutdown(self) -> None:
        await self._yuque_loader.close()

    async def chat(self, question: str, *, model: Optional[str] = None, owner: Optional[str] = None) -> ChatResponse:
        skill_route = route_skill(question)
        skill_id = skill_route.skill_id if skill_route else None
        generation_question = (
            f"[skill_id={skill_id}]\n{skill_route.generation_instruction}\n\n用户问题：{question}"
            if skill_route
            else question
        )
        if model is None and owner is None:
            mode, label = self.runtime_mode()
            logger.info("chat_received mode=%s label=%s question=%r", mode, label, question)
            response = await self._pipeline.run(
                question,
                retrieval_question=question,
                generation_question=generation_question,
                skill_id=skill_id,
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
            return response

        response = await self._run_one(
            question,
            model=model,
            owner=owner,
            skill_id=skill_id,
            generation_question=generation_question,
        )
        return response

    async def chat_stream(
        self, question: str, *, model: Optional[str] = None, owner: Optional[str] = None
    ) -> AsyncIterator[dict[str, Any]]:
        skill_route = route_skill(question)
        skill_id = skill_route.skill_id if skill_route else None
        generation_question = (
            f"[skill_id={skill_id}]\n{skill_route.generation_instruction}\n\n用户问题：{question}"
            if skill_route
            else question
        )
        if model is None and owner is None:
            mode, label = self.runtime_mode()
            logger.info("chat_received mode=%s label=%s question=%r", mode, label, question)
            retrieval, debug, answer_stream = await self._pipeline.run_stream(
                question,
                retrieval_question=question,
                generation_question=generation_question,
                skill_id=skill_id,
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
            yield {"event": "done", "data": response.model_dump()}
            return

        async for event in self._run_one_stream(
            question,
            model=model,
            owner=owner,
            skill_id=skill_id,
            generation_question=generation_question,
        ):
            yield event

    def runtime_mode(self) -> tuple[str, str]:
        if self._embedder is None:
            return "direct_yuque", "语雀直连模式"
        return "rag", "RAG 向量模式"

    async def _run_one(
        self,
        question: str,
        *,
        model: Optional[str],
        owner: Optional[str],
        skill_id: Optional[str],
        generation_question: str,
    ) -> ChatResponse:
        # 语雀作用域与向量检索作用域可能不一致：当 owner 非默认作用域时，强制走直连（embedder=None）。
        scope = self._compute_yuque_scope(owner)
        embedder_for_retriever = self._embedder if scope == settings.yuque_scope else None
        generator = self._build_generator_by_selected_model(model or settings.llm_model)

        yuque_loader = self._build_yuque_loader(scope)
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
        try:
            response = await pipeline.run(
                question,
                retrieval_question=question,
                generation_question=generation_question,
                skill_id=skill_id,
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
        return response

    async def _run_one_stream(
        self,
        question: str,
        *,
        model: Optional[str],
        owner: Optional[str],
        skill_id: Optional[str],
        generation_question: str,
    ) -> AsyncIterator[dict[str, Any]]:
        scope = self._compute_yuque_scope(owner)
        embedder_for_retriever = self._embedder if scope == settings.yuque_scope else None
        generator = self._build_generator_by_selected_model(model or settings.llm_model)

        yuque_loader = self._build_yuque_loader(scope)
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

        completed = False
        try:
            retrieval, debug, answer_stream = await pipeline.run_stream(
                question,
                retrieval_question=question,
                generation_question=generation_question,
                skill_id=skill_id,
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
            yield {"event": "done", "data": response.model_dump()}
        finally:
            if not completed:
                logger.info("chat_stream(dyn) ended_before_done owner=%r scope=%r", owner, scope)
            await yuque_loader.close()

    def _compute_yuque_scope(self, owner: Optional[str]) -> str:
        default_scope = (settings.yuque_scope or "").strip().strip("/")
        if not owner:
            return default_scope
        owner = owner.strip().strip("/")
        if not default_scope or "/" not in default_scope:
            return owner
        _, repo = default_scope.split("/", 1)
        return f"{owner}/{repo}"

    def _build_yuque_loader(self, scope: str) -> YuqueLoader:
        return YuqueLoader(
            token=settings.yuque_token,
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
        integrated = {"yuque_search", "yuque_get_doc", "yuque_list_docs", "yuque_get_toc"}
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
                "status": "integrated" if name in integrated else "available",
                "description": description,
            }
            for name, category, description in tool_items
        ]
        return {
            "enabled": self._mcp_client.enabled,
            "repo_scope": settings.yuque_scope,
            "tools": tools,
        }

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

