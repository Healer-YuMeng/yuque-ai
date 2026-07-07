from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from openai import AsyncOpenAI
from app.core.config import settings
from app.data.mcp_client import MCPClientError, MCPDocMeta, MCPTocNode, YuqueMCPClient
from app.data.yuque_loader import YuqueLoader, YuqueLoaderError, YuqueTocNode
from app.core.logger import get_logger
from app.rag.embedder import Embedder
from app.schemas.chat import SourceItem
from app.storage.vector_store import RetrievedChunk, VectorStore


logger = get_logger(__name__)
IntentType = Literal["directory", "doc_list", "content"]


@dataclass(frozen=True)
class RetrievalResult:
    contexts: List[str]
    sources: List[SourceItem]
    fallback_used: bool
    debug: Dict[str, Any] = field(default_factory=dict)


class Retriever:
    def __init__(
        self,
        *,
        vector_store: VectorStore,
        embedder: Optional[Embedder],
        mcp_client: YuqueMCPClient,
        yuque_loader: Optional[YuqueLoader],
        top_k: int,
        score_threshold: float,
    ) -> None:
        self._vector_store = vector_store
        self._embedder = embedder
        self._mcp_client = mcp_client
        self._yuque_loader = yuque_loader
        self._top_k = top_k
        self._score_threshold = score_threshold
        self._intent_client: Optional[AsyncOpenAI] = None
        intent_key, intent_base = settings.resolve_model_endpoint(settings.intent_llm_model)
        if settings.intent_llm_enabled and intent_key:
            self._intent_client = AsyncOpenAI(
                api_key=intent_key,
                base_url=intent_base or None,
            )

    @property
    def yuque_loader(self) -> Optional[YuqueLoader]:
        return self._yuque_loader

    async def retrieve(
        self,
        question: str,
        *,
        skill_id: Optional[str] = None,
        doc_anchors: Optional[List[tuple[int, Optional[str]]]] = None,
    ) -> RetrievalResult:
        # 只在特定 skill 场景下改变检索策略；其它 skill 仍基于用户原问题检索。
        if skill_id == "stale-detector":
            return await self._retrieve_stale_detector_context(question)
        anchors = doc_anchors or []
        if self._embedder is None:
            logger.info("retrieval_mode=direct_yuque question=%r", question)
            return await self._retrieve_from_yuque(question, skill_id=skill_id, doc_anchors=anchors or None)

        logger.info("retrieval_mode=vector question=%r", question)
        query_embedding = await self._embedder.embed_query(question)
        hits = await self._vector_store.search(query_embedding, self._top_k)
        logger.info("vector_hits=%d top_k=%d", len(hits), self._top_k)

        if anchors:
            anchor_keys = {str(a[0]) for a in anchors}
            anchored_hits = [hit for hit in hits if hit.chunk.doc_id in anchor_keys]
            if anchored_hits:
                strong_hits = [hit for hit in anchored_hits if hit.score >= self._score_threshold]
                picks = strong_hits if strong_hits else anchored_hits[: self._top_k]
                logger.info("vector_anchor_prefilter picks=%d anchor_keys=%s", len(picks), sorted(anchor_keys))
                return RetrievalResult(
                    contexts=[hit.chunk.text for hit in picks],
                    sources=[self._to_source(hit) for hit in picks],
                    fallback_used=False,
                    debug={"anchor_doc_ids": sorted(anchor_keys), "retrieval_mode": "vector_anchor"},
                )
            anchor_pull = await self._retrieve_from_doc_anchors(anchors)
            if anchor_pull is not None and anchor_pull.contexts:
                logger.info("vector_anchor_openapi_fallback contexts=%d", len(anchor_pull.contexts))
                return anchor_pull

        strong_hits = [hit for hit in hits if hit.score >= self._score_threshold]
        if strong_hits:
            logger.info("vector_strong_hits=%d threshold=%.3f", len(strong_hits), self._score_threshold)
            return RetrievalResult(
                contexts=[hit.chunk.text for hit in strong_hits],
                sources=[self._to_source(hit) for hit in strong_hits],
                fallback_used=False,
            )

        mcp_results = await self._mcp_client.search(question)
        if mcp_results:
            logger.info("vector_to_mcp_fallback results=%d", len(mcp_results))
            contexts = [item.snippet or item.title for item in mcp_results[: self._top_k]]
            sources = [
                SourceItem(title=item.title, url=item.url, source_type="mcp", snippet=item.snippet)
                for item in mcp_results[: self._top_k]
            ]
            return RetrievalResult(contexts=contexts, sources=sources, fallback_used=True)

        logger.info("vector_weak_hits_returned count=%d", len(hits))
        return RetrievalResult(
            contexts=[hit.chunk.text for hit in hits],
            sources=[self._to_source(hit) for hit in hits],
            fallback_used=False,
        )

    async def _retrieve_stale_detector_context(self, question: str) -> RetrievalResult:
        """
        stale-detector（只读）检索阶段：取当前知识库文档列表，并把 updated_at 注入 contexts。
        生成阶段会据此判断哪些可能过期/需要更新。
        """
        if self._yuque_loader is None:
            return RetrievalResult(
                contexts=[],
                sources=[],
                fallback_used=False,
                debug={
                    "retrieval_mode": "stale_detector",
                    "skill_id": "stale-detector",
                    "reason": "yuque_loader_missing",
                },
            )

        scope = getattr(self._yuque_loader, "_scope", "")
        if not scope:
            return RetrievalResult(
                contexts=[],
                sources=[],
                fallback_used=False,
                debug={
                    "retrieval_mode": "stale_detector",
                    "skill_id": "stale-detector",
                    "reason": "yuque_scope_missing",
                },
            )

        try:
            docs = await self._yuque_loader.list_docs(book=scope, offset=0, limit=60)
        except Exception as exc:
            logger.warning("stale_detector_list_docs_failed scope=%r err=%s", scope, exc)
            docs = []

        lines: List[str] = []
        for doc in docs:
            updated = (getattr(doc, "updated_at", "") or "").strip()
            title = (getattr(doc, "title", "") or "").strip()
            if not title:
                continue
            # 为了给生成阶段足够信号，同时尽量保持 contexts 可读。
            lines.append(f"- {title}（updated_at: {updated or 'unknown'}）")

        contexts = [
            "疑似过期候选文档列表（仅基于 updated_at 的元信息，不保证最终结论）：\n"
            + "\n".join(lines)
        ] if lines else []

        sources = [
            SourceItem(
                title="语雀文档列表（stale-detector）",
                url="",
                source_type="yuque",
                snippet=str((lines[:8] or ["无候选文档"]))[:200],
            )
        ] if lines else []

        return RetrievalResult(
            contexts=contexts,
            sources=sources,
            fallback_used=False,
            debug={
                "retrieval_mode": "stale_detector",
                "skill_id": "stale-detector",
                "question": question,
                "docs_count": len(docs),
                "contexts_count": len(contexts),
            },
        )

    async def _retrieve_from_doc_anchors(
        self, anchors: List[tuple[int, Optional[str]]]
    ) -> Optional[RetrievalResult]:
        """按语雀 doc_id（及可选 slug）拉取正文，用于用户显式选中文档后的检索锚定。"""
        if not anchors or self._yuque_loader is None:
            return None
        scope = (getattr(self._yuque_loader, "_scope", "") or "").strip()
        if not scope:
            return None
        contexts: List[str] = []
        sources: List[SourceItem] = []
        for doc_id, slug in anchors:
            resolved = None
            keys: List[str] = [str(doc_id)]
            slug_tail = (slug or "").strip()
            if slug_tail and slug_tail not in keys:
                keys.append(slug_tail)
            for key in keys:
                try:
                    resolved = await self._yuque_loader.get_doc(book=scope, id_or_slug=key)
                    break
                except YuqueLoaderError:
                    continue
            if resolved is None:
                continue
            body = str(getattr(resolved, "body", "") or "")
            title = str(getattr(resolved, "title", "") or "")
            url = str(getattr(resolved, "url", "") or "")
            did = str(getattr(resolved, "doc_id", "") or doc_id).strip() or None
            contexts.append(body[:4000] if body else title)
            sources.append(
                SourceItem(
                    title=title,
                    url=url,
                    source_type="yuque",
                    snippet=(body or title)[:200],
                    doc_id=did,
                )
            )
        if not contexts:
            return None
        return RetrievalResult(
            contexts=contexts,
            sources=sources,
            fallback_used=False,
            debug={
                "retrieval_mode": "yuque_anchor",
                "anchor_doc_ids": [a[0] for a in anchors],
            },
        )

    async def _retrieve_from_yuque(
        self, question: str, *, skill_id: Optional[str] = None, doc_anchors: Optional[List[tuple[int, Optional[str]]]] = None
    ) -> RetrievalResult:
        if doc_anchors:
            anchored = await self._retrieve_from_doc_anchors(doc_anchors)
            if anchored is not None and anchored.contexts:
                return anchored

        if settings.force_mcp_fallback:
            logger.info("force_mcp_fallback_enabled -> mcp_or_empty")
            result = await self._retrieve_from_mcp_or_empty(question, skill_id=skill_id)
            result.debug.setdefault("forced_mcp", True)
            return result

        if self._yuque_loader is None:
            logger.info("direct_yuque_missing_loader -> mcp_or_empty")
            return await self._retrieve_from_mcp_or_empty(question, skill_id=skill_id)

        intent, intent_source = await self._classify_intent(question, skill_id=skill_id)
        logger.info("direct_yuque_intent intent=%s source=%s", intent, intent_source)

        if intent == "doc_list":
            logger.info("direct_yuque_detected_doc_list_question")
            docs_result = await self._retrieve_doc_list_context()
            if docs_result is not None:
                logger.info("direct_yuque_doc_list_context_ok contexts=%d", len(docs_result.contexts))
                return docs_result

        if intent == "directory":
            logger.info("direct_yuque_detected_directory_question")
            toc_result = await self._retrieve_directory_context(question)
            if toc_result is not None:
                logger.info("direct_yuque_directory_context_ok contexts=%d", len(toc_result.contexts))
                return toc_result

        queries = self._build_search_queries(question)
        logger.info("direct_yuque_search_queries=%s", queries)
        try:
            hits = []
            for query in queries:
                hits = await self._yuque_loader.search_docs(query)
                logger.info("yuque_search query=%r hits=%d", query, len(hits))
                if hits:
                    break
        except YuqueLoaderError:
            logger.warning("yuque_search_failed")
            return await self._retrieve_from_mcp_or_empty(question, skill_id=skill_id)

        contexts: List[str] = []
        sources: List[SourceItem] = []
        selected_hits = []
        tasks = []
        for hit in hits[: self._top_k]:
            identifier = str(hit.doc_id or hit.slug or "")
            if hit.book_id is None or not identifier:
                continue
            selected_hits.append((hit, identifier))
            tasks.append(self._yuque_loader.get_doc(book=hit.book_id, id_or_slug=identifier))

        results = await asyncio.gather(*tasks, return_exceptions=True) if tasks else []
        docs_fetched = 0
        for (hit, _identifier), result in zip(selected_hits, results):
            if isinstance(result, Exception):
                continue
            doc = result
            docs_fetched += 1
            contexts.append(doc.body[:4000] if doc.body else hit.summary)
            doc_id_val = str(getattr(doc, "doc_id", "") or hit.doc_id or "").strip() or None
            sources.append(
                SourceItem(
                    title=doc.title or hit.title,
                    url=doc.url or hit.url,
                    source_type="yuque",
                    snippet=(doc.body or hit.summary)[:200],
                    doc_id=doc_id_val,
                )
            )

        if contexts:
            logger.info(
                "direct_yuque_docs_fetched=%d contexts=%d sources=%d",
                docs_fetched,
                len(contexts),
                len(sources),
            )
            return RetrievalResult(contexts=contexts, sources=sources, fallback_used=False)

        docs_title_fallback = await self._retrieve_from_docs_title_match(question)
        if docs_title_fallback is not None:
            logger.info("direct_yuque_docs_title_fallback_ok sources=%d", len(docs_title_fallback.sources))
            return docs_title_fallback

        toc_fallback = await self._retrieve_from_toc_titles(question)
        if toc_fallback is not None:
            logger.info("direct_yuque_toc_title_fallback_ok sources=%d", len(toc_fallback.sources))
            return toc_fallback
        logger.info("direct_yuque_no_context -> mcp_or_empty")
        return await self._retrieve_from_mcp_or_empty(question, skill_id=skill_id)

    async def _retrieve_doc_list_context(self) -> Optional[RetrievalResult]:
        if self._yuque_loader is None:
            return None
        scope = getattr(self._yuque_loader, "_scope", "")
        if not scope:
            return None
        try:
            docs = await self._yuque_loader.list_docs(book=scope, offset=0, limit=50)
        except YuqueLoaderError:
            logger.warning("yuque_docs_list_failed scope=%r", scope)
            return None
        if not docs:
            logger.info("yuque_docs_list_empty")
            return None
        lines = [f"- {doc.title}" for doc in docs[:30]]
        context = "\n".join(lines)
        return RetrievalResult(
            contexts=[f"当前知识库文档列表如下：\n{context}"],
            sources=[
                SourceItem(
                    title="知识库文档列表",
                    url=docs[0].url if docs else "",
                    source_type="yuque",
                    snippet=context[:200],
                )
            ],
            fallback_used=False,
        )

    async def _retrieve_from_docs_title_match(self, question: str) -> Optional[RetrievalResult]:
        if self._yuque_loader is None:
            return None
        scope = getattr(self._yuque_loader, "_scope", "")
        if not scope:
            return None
        try:
            docs = await self._yuque_loader.list_docs(book=scope, offset=0, limit=100)
        except YuqueLoaderError:
            return None
        keywords = self._keyword_parts(question)
        candidates = [
            doc
            for doc in docs
            if any(keyword and keyword in doc.title for keyword in keywords)
        ][: self._top_k]
        if not candidates:
            logger.info("yuque_docs_title_fallback_no_match keywords=%s", keywords)
            return None

        tasks = [self._yuque_loader.get_doc(book=scope, id_or_slug=str(doc.id or doc.slug)) for doc in candidates]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        contexts: List[str] = []
        sources: List[SourceItem] = []
        for doc_meta, result in zip(candidates, results):
            if isinstance(result, Exception):
                continue
            contexts.append(result.body[:4000] if result.body else doc_meta.title)
            sources.append(
                SourceItem(
                    title=result.title or doc_meta.title,
                    url=result.url or doc_meta.url,
                    source_type="yuque",
                    snippet=(result.body or doc_meta.title)[:200],
                )
            )
        if not contexts:
            return None
        return RetrievalResult(contexts=contexts, sources=sources, fallback_used=False)

    async def _retrieve_directory_context(self, question: str) -> Optional[RetrievalResult]:
        if self._yuque_loader is None:
            return None
        scope = getattr(self._yuque_loader, "_scope", "")
        if not scope:
            return None
        try:
            toc_nodes = await self._yuque_loader.get_book_toc(book=scope)
        except YuqueLoaderError:
            logger.warning("yuque_toc_failed scope=%r", scope)
            return None

        normalized = (
            question.replace("有什么", "")
            .replace("有哪些", "")
            .replace("目录", "")
            .replace("子目录", "")
            .replace("子文档", "")
            .strip()
        )
        normalized = re.sub(r"[？?。！，、\s]+", "", normalized)
        generic_subdir = bool(
            re.fullmatch(
                r"(有)?子(吗)?|(有)?(子)?目录(吗)?|(有)?子文档(吗)?|(有)?(子)?章节(吗)?",
                normalized,
            )
        )
        if (not normalized) or (len(normalized) <= 2) or generic_subdir:
            root_nodes = [node for node in toc_nodes if not node.parent_uuid]
            if not root_nodes:
                root_nodes = [node for node in toc_nodes if node.level <= 1] or toc_nodes[:20]

            children_map: dict[str, List[YuqueTocNode]] = {}
            for node in toc_nodes:
                children_map.setdefault(node.parent_uuid, []).append(node)

            lines: List[str] = []
            max_roots = 12
            max_children_each = 8
            for root in root_nodes[:max_roots]:
                lines.append(self._format_toc_line(root))
                for child in (children_map.get(root.uuid, []) or [])[:max_children_each]:
                    lines.append(self._format_toc_line(child))

            context = "\n".join(lines) if lines else ""
            logger.info("yuque_toc_root_list roots=%d lines=%d", len(root_nodes[:max_roots]), len(lines))
            return RetrievalResult(
                contexts=[f"当前知识库目录如下：\n{context}"],
                sources=[
                    SourceItem(
                        title="知识库目录",
                        url=root_nodes[0].url if root_nodes else "",
                        source_type="yuque",
                        snippet=context[:200],
                    )
                ],
                fallback_used=False,
            )

        target = self._best_toc_match(toc_nodes, normalized)
        if target is None:
            logger.info("yuque_toc_no_match query=%r", normalized)
            return None
        subtree = self._subtree_from_uuid(toc_nodes, target.uuid)
        lines = [self._format_toc_line(target)] + [self._format_toc_line(node) for node in subtree[:20]]
        context = "\n".join(lines)
        logger.info("yuque_toc_match title=%r subtree=%d", target.title, len(subtree))
        return RetrievalResult(
            contexts=[f"与问题相关的目录结构如下：\n{context}"],
            sources=[SourceItem(title=target.title, url=target.url, source_type="yuque", snippet=context[:200])],
            fallback_used=False,
        )

    async def _retrieve_from_toc_titles(self, question: str) -> Optional[RetrievalResult]:
        if self._yuque_loader is None:
            return None
        scope = getattr(self._yuque_loader, "_scope", "")
        if not scope:
            return None
        try:
            toc_nodes = await self._yuque_loader.get_book_toc(book=scope)
        except YuqueLoaderError:
            return None
        keywords = self._keyword_parts(question)
        matches = [
            node for node in toc_nodes if node.doc_id is not None and any(keyword and keyword in node.title for keyword in keywords)
        ][: self._top_k]
        if not matches:
            logger.info("yuque_toc_title_fallback_no_match keywords=%s", keywords)
            return None

        context = "\n".join(self._format_toc_line(node) for node in matches)
        logger.info("yuque_toc_title_fallback_matches=%d", len(matches))
        return RetrievalResult(
            contexts=[f"从知识库目录标题匹配到以下候选文档：\n{context}"],
            sources=[
                SourceItem(title=node.title, url=node.url, source_type="yuque", snippet=node.title)
                for node in matches
            ],
            fallback_used=False,
        )

    @staticmethod
    def _wants_rich_doc_inventory(question: str) -> bool:
        """用户明确要求层级/缩进/字数/类型等时的增强问法（与 doc_list 合并拉 TOC + list）。"""
        t = (question or "").strip()
        # 注意：不要用单独的「图片」——用户常问「某图下面的文字」等正文问题，会误触合并清单而拿不到 get_doc 正文。
        image_meta = any(
            k in t
            for k in (
                "图片数",
                "图片数量",
                "几张图",
                "多少张图",
                "图数量",
            )
        )
        return image_meta or any(
            k in t
            for k in (
                "结构",
                "层级",
                "层次",
                "缩进",
                "父文档",
                "子文档",
                "目录树",
                "大纲",
                "分明",
                "一目了然",
                "字数",
                "类型",
                "篇幅",
                "可见",
                "不可见",
            )
        )

    @staticmethod
    def _wants_document_visual_content(question: str) -> bool:
        """
        用户关心「某篇/某主题文档里的插图长什么样、有哪些图」，需要正文与插图代理，不是知识库 TOC+清单表。
        与 doc_list / combined_inventory / 仅元数据「图片数」类问题区分。
        """
        t = (question or "").strip()
        if not t:
            return False
        if not any(k in t for k in ("图片", "插图", "截图", "配图")):
            return False
        if any(
            k in t
            for k in (
                "图片数",
                "图片数量",
                "几张图",
                "多少张图",
                "图数量",
                "各文档图片",
                "统计表",
            )
        ):
            return False
        if any(
            k in t
            for k in (
                "目录树",
                "文档清单",
                "都有哪些文档",
                "知识库有哪些文档",
                "文档列表",
                "全部文档",
                "哪些文档",
                "几个文档",
                "多少篇文档",
            )
        ):
            return False
        return True

    @staticmethod
    def _format_mcp_combined_inventory_context(docs: List[MCPDocMeta], toc_nodes: List[MCPTocNode]) -> str:
        lines: List[str] = []
        lines.append("【合并知识库清单｜数据来自 yuque_get_toc + yuque_list_docs，请勿编造表中不存在的数字】")
        lines.append("")
        lines.append("### 输出要求（请在最终回答中落实）")
        lines.append("- 用 Markdown：先输出**带层级缩进的目录树**（level 每增加 1，子级多两个空格或嵌套列表项）。")
        lines.append("- 再输出**文档统计表**：列含 doc_id、标题、字数/正文长度、图片数、可见性；字段无数据时写「未提供」，勿猜。")
        lines.append("- 目录树与表格中同一 doc_id 应互相对应。")
        lines.append("")
        lines.append("## 一、目录树（yuque_get_toc）")
        if toc_nodes:
            for node in toc_nodes[:120]:
                indent = "  " * max(int(node.level), 0)
                vis = ""
                if node.visible is False:
                    vis = "（不可见）"
                elif node.visible is True:
                    vis = "（可见）"
                did = f" `doc_id={node.doc_id}`" if node.doc_id else ""
                lines.append(f"{indent}- {node.title}{vis}{did}")
        else:
            lines.append("（无 TOC 数据）")
        lines.append("")
        lines.append("## 二、文档清单（yuque_list_docs）")
        lines.append("| doc_id | 标题 | 字数/长度 | 图片数 | 类型 | 可见/公开 |")
        lines.append("|---|---|---|---|---|---|")
        if docs:
            for d in docs[:120]:
                wc = "未提供"
                if d.word_count is not None:
                    wc = str(d.word_count)
                elif d.body_length is not None:
                    wc = str(d.body_length)
                img = str(d.image_count) if d.image_count is not None else "未提供"
                dtype = d.doc_type or "未提供"
                vis_parts: List[str] = []
                if d.visible is not None:
                    vis_parts.append("visible=" + ("是" if d.visible else "否"))
                if d.public is not None:
                    vis_parts.append("public=" + ("是" if d.public else "否"))
                vis_s = "；".join(vis_parts) if vis_parts else "未提供"
                title = (d.title or "").replace("|", "\\|")
                lines.append(f"| {d.doc_id} | {title} | {wc} | {img} | {dtype} | {vis_s} |")
        else:
            lines.append("| — | （无 list_docs 数据） | — | — | — | — |")
        return "\n".join(lines)

    async def _retrieve_mcp_combined_doc_inventory(
        self, question: str, *, intent: IntentType, intent_source: str
    ) -> Optional[RetrievalResult]:
        """并行拉取 TOC + 文档列表，生成单一结构化上下文，避免 auto_router 只调一个工具导致信息不全。"""
        if not self._mcp_client.enabled:
            return None
        try:
            docs, toc_nodes = await asyncio.gather(
                self._mcp_client.list_docs(),
                self._mcp_client.get_toc(),
            )
        except Exception as exc:
            logger.warning("mcp_combined_inventory_gather_failed err=%s", exc)
            return None
        if not docs and not toc_nodes:
            return None
        body = self._format_mcp_combined_inventory_context(docs, toc_nodes)
        body = body[:14000]
        return RetrievalResult(
            contexts=[body],
            sources=[
                SourceItem(
                    title="知识库目录+文档清单（MCP 合并）",
                    url="",
                    source_type="mcp",
                    snippet=body[:220],
                )
            ],
            fallback_used=True,
            debug={
                "retrieval_mode": "mcp_fallback",
                "mcp_route": "combined_inventory",
                "intent": intent,
                "intent_source": intent_source,
                "mcp_used_tools": [self._mcp_client.list_docs_tool, self._mcp_client.get_toc_tool],
                "mcp_list_docs_count": len(docs),
                "mcp_toc_node_count": len(toc_nodes),
                "mcp_combined_inventory": True,
            },
        )

    async def _retrieve_from_mcp_or_empty(self, question: str, *, skill_id: Optional[str] = None) -> RetrievalResult:
        intent, intent_source = await self._classify_intent(question, skill_id=skill_id)
        if (
            self._mcp_client.enabled
            and (intent == "doc_list" or self._wants_rich_doc_inventory(question))
            and not self._wants_document_visual_content(question)
        ):
            combined = await self._retrieve_mcp_combined_doc_inventory(
                question, intent=intent, intent_source=intent_source
            )
            if combined is not None:
                return combined
        mcp_queries: List[str] = []
        if settings.auto_mcp_tool_router:
            auto_result = await self._try_auto_mcp_tool_route(question, intent=intent, intent_source=intent_source)
            if auto_result is not None:
                return auto_result
        if intent == "directory":
            toc_nodes = await self._mcp_client.get_toc()
            if toc_nodes:
                lines = [f"{'  ' * max(node.level - 1, 0)}- {node.title}" for node in toc_nodes[:40]]
                context = "当前知识库目录如下：\n" + "\n".join(lines)
                return RetrievalResult(
                    contexts=[context],
                    sources=[
                        SourceItem(
                            title="知识库目录（MCP）",
                            url="",
                            source_type="mcp",
                            snippet="\n".join(lines[:8])[:200],
                        )
                    ],
                    fallback_used=True,
                    debug={
                        "retrieval_mode": "mcp_fallback",
                        "mcp_route": "toc",
                        "intent": intent,
                        "intent_source": intent_source,
                        "mcp_queries": [],
                        "mcp_used_tools": [self._mcp_client.get_toc_tool],
                        "mcp_search_result_count": 0,
                        "mcp_result_count": len(toc_nodes),
                        "mcp_get_doc_requested": 0,
                        "mcp_get_doc_fetched": 0,
                    },
                )

        if intent == "doc_list":
            docs = await self._mcp_client.list_docs()
            if docs:
                lines = [f"- {doc.title}" for doc in docs[:40]]
                context = "当前知识库文档列表如下：\n" + "\n".join(lines)
                return RetrievalResult(
                    contexts=[context],
                    sources=[
                        SourceItem(
                            title="知识库文档列表（MCP）",
                            url="",
                            source_type="mcp",
                            snippet="\n".join(lines[:8])[:200],
                        )
                    ],
                    fallback_used=True,
                    debug={
                        "retrieval_mode": "mcp_fallback",
                        "mcp_route": "list_docs",
                        "intent": intent,
                        "intent_source": intent_source,
                        "mcp_queries": [],
                        "mcp_used_tools": [self._mcp_client.list_docs_tool],
                        "mcp_search_result_count": 0,
                        "mcp_result_count": len(docs),
                        "mcp_get_doc_requested": 0,
                        "mcp_get_doc_fetched": 0,
                    },
                )

        for query in self._build_search_queries(question):
            mcp_queries.append(query)
            try:
                mcp_results = await self._mcp_client.search(query)
            except MCPClientError as exc:
                logger.warning("mcp_search_failed query=%r err=%s", query, exc)
                continue
            logger.info("mcp_search query=%r results=%d", query, len(mcp_results))
            if not mcp_results:
                continue
            logger.info("mcp_fallback_ok results=%d", len(mcp_results))
            selected = mcp_results[: self._top_k]
            doc_tasks = [self._mcp_client.get_doc(item.doc_id) for item in selected if item.doc_id]
            doc_bodies = await asyncio.gather(*doc_tasks, return_exceptions=True) if doc_tasks else []
            body_map: dict[str, str] = {}
            body_idx = 0
            for item in selected:
                if not item.doc_id:
                    continue
                body = doc_bodies[body_idx]
                body_idx += 1
                if isinstance(body, Exception):
                    continue
                text = (body or "").strip()
                if text:
                    body_map[item.doc_id] = self._strip_html(text)
            logger.info("mcp_get_doc fetched=%d requested=%d", len(body_map), len(doc_tasks))
            # 把文档标题显式拼进上下文，避免 LLM 因为上下文缺少“标题词”而误判未命中。
            contexts = [
                f"文档标题：{item.title}\n\n{(body_map.get(item.doc_id) or item.snippet or item.title)[:3800]}"
                for item in selected
            ]
            sources = [
                SourceItem(
                    title=item.title,
                    url=item.url,
                    source_type="mcp",
                    snippet=(body_map.get(item.doc_id) or item.snippet or item.title)[:200],
                    doc_id=item.doc_id,
                )
                for item in selected
            ]
            return RetrievalResult(
                contexts=contexts,
                sources=sources,
                fallback_used=True,
                debug={
                    "retrieval_mode": "mcp_fallback",
                    "mcp_route": "search",
                    "intent": intent,
                    "intent_source": intent_source,
                    "mcp_queries": mcp_queries,
                    "mcp_used_tools": [self._mcp_client.search_tool, self._mcp_client.get_doc_tool],
                    "mcp_search_result_count": len(mcp_results),
                    "mcp_result_count": len(selected),
                    "mcp_get_doc_requested": len(doc_tasks),
                    "mcp_get_doc_fetched": len(body_map),
                },
            )

        # MCP 搜索未命中时，回退到 docs 标题匹配，避免口语问法导致 search=0
        try:
            docs = await self._mcp_client.list_docs()
        except MCPClientError as exc:
            logger.warning("mcp_list_docs_title_fallback_failed err=%s", exc)
            docs = []
        keywords = self._keyword_parts(question)
        normalized_question = re.sub(r"\s+", "", question)
        matched_docs = []
        for doc in docs:
            title = (doc.title or "").strip()
            if not title:
                continue
            if any(keyword and (keyword in title or title in keyword) for keyword in keywords):
                matched_docs.append(doc)
                continue
            # 双向兜底：问题包含标题或标题包含问题核心片段
            if title in normalized_question or normalized_question in title:
                matched_docs.append(doc)
        matched_docs = matched_docs[: self._top_k]
        if matched_docs:
            doc_tasks = [self._mcp_client.get_doc(doc.doc_id or doc.slug) for doc in matched_docs]
            doc_bodies = await asyncio.gather(*doc_tasks, return_exceptions=True)
            contexts: List[str] = []
            sources: List[SourceItem] = []
            fetched = 0
            for doc, body in zip(matched_docs, doc_bodies):
                if isinstance(body, Exception):
                    continue
                text = self._strip_html((body or "").strip())
                if not text:
                    continue
                fetched += 1
                # 把文档标题显式拼进上下文，增强“按文档名提问”的命中率。
                contexts.append(f"文档标题：{doc.title}\n\n{text[:4000]}")
                sources.append(
                    SourceItem(
                        title=doc.title,
                        url=doc.url,
                        source_type="mcp",
                        snippet=text[:200],
                        doc_id=doc.doc_id,
                    )
                )
            if contexts:
                logger.info("mcp_docs_title_fallback_ok matched=%d fetched=%d", len(matched_docs), fetched)
                return RetrievalResult(
                    contexts=contexts,
                    sources=sources,
                    fallback_used=True,
                    debug={
                        "retrieval_mode": "mcp_fallback",
                        "mcp_route": "title_fallback",
                        "intent": intent,
                        "intent_source": intent_source,
                        "mcp_queries": mcp_queries,
                        "mcp_used_tools": [self._mcp_client.list_docs_tool, self._mcp_client.get_doc_tool],
                        "mcp_search_result_count": 0,
                        "mcp_result_count": len(matched_docs),
                        "mcp_get_doc_requested": len(doc_tasks),
                        "mcp_get_doc_fetched": fetched,
                        "mcp_title_fallback": True,
                    },
                )

        logger.info("mcp_fallback_empty")
        return RetrievalResult(
            contexts=[],
            sources=[],
            fallback_used=False,
            debug={
                "retrieval_mode": "mcp_fallback",
                "mcp_route": "search_empty",
                "intent": intent,
                "intent_source": intent_source,
                "mcp_queries": mcp_queries,
                "mcp_used_tools": [self._mcp_client.search_tool],
                "mcp_search_result_count": 0,
                "mcp_result_count": 0,
                "mcp_get_doc_requested": 0,
                "mcp_get_doc_fetched": 0,
            },
        )

    async def _try_auto_mcp_tool_route(
        self, question: str, *, intent: IntentType, intent_source: str
    ) -> Optional[RetrievalResult]:
        if not self._intent_client:
            return None
        # 兼容测试/最小实现：如果 mcp_client 不提供自动路由所需的字段/能力，则跳过。
        tool_list = getattr(self._mcp_client, "read_tools", None)
        call_raw = getattr(self._mcp_client, "call_raw", None)
        repo_id = getattr(self._mcp_client, "repo_id", "")
        if not tool_list or not callable(call_raw):
            return None
        tool_desc = ", ".join(tool_list)
        prompt = (
            "你是 MCP 工具路由器。请从可用只读工具中选择最合适的一项，并返回 JSON。"
            "不要解释。\n"
            f"可用工具: {tool_desc}\n"
            f"默认 repo_id: {repo_id}\n"
            "JSON 格式: {\"tool\":\"...\",\"arguments\":{...}}\n"
            f"用户问题: {question}"
        )
        try:
            resp = await self._intent_client.chat.completions.create(
                model=settings.intent_llm_model,
                temperature=0,
                extra_body={"enable_thinking": False},
                messages=[
                    {"role": "system", "content": "只输出合法 JSON，不要 markdown。"},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = (resp.choices[0].message.content or "").strip()
            plan = json.loads(raw)
        except Exception as exc:
            logger.info("mcp_auto_router_skip error=%s", exc)
            return None

        tool = str(plan.get("tool") or "").strip()
        args = plan.get("arguments") if isinstance(plan.get("arguments"), dict) else {}
        if tool not in tool_list:
            return None
        if "repo_id" not in args and tool in {
            "yuque_get_book",
            self._mcp_client.list_docs_tool,
            self._mcp_client.get_toc_tool,
            self._mcp_client.get_doc_tool,
        }:
            args["repo_id"] = repo_id

        # 兼容：yuque_search 在 MCP 侧要求 type 参数（doc/repo）
        if tool == self._mcp_client.search_tool:
            # 部分 mcp-server 会要求 repo_id（用于限定检索范围）
            if "repo_id" not in args and repo_id:
                args["repo_id"] = repo_id
            if "type" not in args:
                # 默认按文档检索（与我们 mcp_client.search 的行为一致）
                args["type"] = "doc"

        # 兼容：LLM 可能把“标题”当成 yuque_get_doc 的 doc_id 传入，导致 doc not found。
        # 如果 doc_id 看起来不是数字，就先用 MCP search 把标题映射到真实 doc_id。
        if tool == self._mcp_client.get_doc_tool and "doc_id" in args:
            raw_doc_id = str(args.get("doc_id") or "").strip()
            if raw_doc_id and not raw_doc_id.isdigit():
                try:
                    search_hits = await self._mcp_client.search(raw_doc_id)
                    if search_hits:
                        # 优先标题包含匹配，否则用第一个结果兜底
                        best = next((h for h in search_hits[:5] if raw_doc_id in (h.title or "")), None) or search_hits[0]
                        if best.doc_id:
                            args["doc_id"] = str(best.doc_id)
                except Exception as exc:
                    logger.info("mcp_auto_router_docid_map_failed doc_id=%r err=%s", raw_doc_id, exc)
        try:
            payload = await call_raw(tool, args)
        except Exception as exc:
            logger.info("mcp_auto_router_call_failed tool=%s error=%s", tool, exc)
            return None

        # 如果搜索结果为空（如 []），不要直接返回，让后续 title_fallback 尝试基于标题拉正文
        if isinstance(payload, list) and not payload:
            return None

        # 如果 auto_router 走的是 yuque_search，但搜索返回的标题不匹配问题核心关键词，
        # 认为这次命中不可靠，继续让后续逻辑基于标题拉正文兜底。
        if tool == self._mcp_client.search_tool and isinstance(payload, list):
            keywords = self._keyword_parts(question) or []
            titles = [(getattr(it, "title", None) or (it.get("title") if isinstance(it, dict) else "")) for it in payload]
            if keywords:
                matched = any(any(kw and kw in (t or "") for kw in keywords) for t in titles)
                if not matched:
                    return None

        sources: List[SourceItem] = []
        context: str = ""
        search_n = 0

        if tool == self._mcp_client.search_tool:
            parsed_hits = self._mcp_client._parse_search_results(payload)
            parsed_hits = [
                h
                for h in parsed_hits
                if (h.title or "").strip() or (h.snippet or "").strip()
            ][: self._top_k]
            if not parsed_hits:
                return None
            blocks: List[str] = []
            for h in parsed_hits:
                body = (h.snippet or h.title or "").strip()
                blocks.append(f"文档标题：{h.title}\n\n{body[:2000]}")
            context = "\n\n---\n\n".join(blocks)[:4000]
            search_n = len(parsed_hits)
            sources = [
                SourceItem(
                    title=(h.title or "未命名文档").strip(),
                    url=(h.url or "").strip() or None,
                    source_type="mcp",
                    snippet=(h.snippet or h.title or "")[:200],
                    doc_id=(h.doc_id or "").strip() or None,
                )
                for h in parsed_hits
            ]
        elif isinstance(payload, dict) and tool == self._mcp_client.get_doc_tool:
            title = str(payload.get("title") or "").strip() or "语雀文档"
            url_raw = str(payload.get("url") or "").strip()
            body = str(payload.get("body") or payload.get("content") or "").strip()
            doc_id_raw = str(payload.get("id") or args.get("doc_id") or "").strip()
            context = (body[:4000] if body else self._to_plain_text(payload)[:4000]) or "未查询到可用内容。"
            sources = [
                SourceItem(
                    title=title,
                    url=url_raw or None,
                    source_type="mcp",
                    snippet=(body or title)[:200],
                    doc_id=doc_id_raw or None,
                )
            ]
        else:
            context = (self._to_plain_text(payload)[:4000] or "未查询到可用内容。").strip()
            if isinstance(payload, list) and payload and isinstance(payload[0], dict):
                for item in payload[: self._top_k]:
                    if not isinstance(item, dict):
                        continue
                    t = str(item.get("title") or "").strip()
                    if not t:
                        continue
                    u = str(item.get("url") or "").strip() or None
                    did = str(item.get("id") or item.get("doc_id") or "").strip() or None
                    sources.append(
                        SourceItem(
                            title=t,
                            url=u,
                            source_type="mcp",
                            snippet=str(item.get("snippet") or item.get("summary") or t)[:200],
                            doc_id=did or None,
                        )
                    )
            if not sources:
                snippet = context[:200] if context else ""
                sources = [
                    SourceItem(
                        title="知识库检索摘要",
                        url=None,
                        source_type="mcp",
                        snippet=snippet,
                    )
                ]

        if context.strip() in ("", "[]"):
            return None
        # 如果 MCP 明确表示没找到文档，则让上层继续走其它检索兜底
        lower = context.lower()
        if "doc not found" in lower or "resource does not exist" in lower or "not found" in lower:
            return None
        return RetrievalResult(
            contexts=[context],
            sources=sources,
            fallback_used=True,
            debug={
                "retrieval_mode": "mcp_fallback",
                "mcp_route": "auto_router",
                "intent": intent,
                "intent_source": intent_source,
                "mcp_queries": [],
                "mcp_used_tools": [tool],
                "mcp_search_result_count": search_n,
                "mcp_result_count": len(sources),
                "mcp_get_doc_requested": 0,
                "mcp_get_doc_fetched": 0,
                "mcp_auto_router": True,
                "mcp_auto_arguments": args,
            },
        )

    @staticmethod
    def _to_source(hit: RetrievedChunk) -> SourceItem:
        return SourceItem(
            title=hit.chunk.title,
            url=hit.chunk.url,
            source_type="vector",
            snippet=hit.chunk.text[:200],
            score=hit.score,
            doc_id=hit.chunk.doc_id,
        )

    @staticmethod
    def _is_directory_question(question: str) -> bool:
        keywords = ("目录", "章节", "结构", "大纲", "子文档", "文档树")
        return any(keyword in question for keyword in keywords)

    @staticmethod
    def _is_doc_list_question(question: str) -> bool:
        if Retriever._wants_document_visual_content(question):
            return False
        keywords = ("文档列表", "有哪些文档", "有哪些内容", "都有哪些文档", "知识库文档", "全部文档")
        if any(keyword in question for keyword in keywords):
            return True
        normalized = re.sub(r"\s+", "", question)
        if "文档" not in normalized:
            return False
        # 放宽识别：覆盖“有什么文档/知识库里有什么文档/文档都有什么”等口语问法
        return any(token in normalized for token in ("哪些", "列表", "全部", "有什么", "有啥", "都有什么", "里面有什么"))

    @staticmethod
    def _build_search_queries(question: str) -> List[str]:
        queries: List[str] = []
        compact = re.sub(r"[？?。！，、\s]", " ", question).strip()
        for candidate in [Retriever._extract_core_phrase(question), compact, question.strip()]:
            candidate = (candidate or "").strip()
            if candidate and candidate not in queries:
                queries.append(candidate)
        return queries

    @staticmethod
    def _extract_core_phrase(question: str) -> str:
        text = question.strip()
        # 针对“X 是什么内容 / X 是什么 / X 的含义”等问法做剥离
        text = re.sub(r"(是什么内容|是什么|什么意思|含义是什么|怎么理解)$", " ", text)
        # 针对“X 里面有什么 / X 里有什么”做剥离
        text = re.sub(r"(里面|里|中的|中)有(什么|啥)$", " ", text)
        replacements = [
            "什么是",
            "是什么",
            "请问",
            "有没有",
            "有哪些",
            "有什么",
            "有啥",
            "里面",
            "里",
            "吗",
            "呢",
            "一下",
            "介绍",
            "说明",
            "内容",
            "简要概述",
            "概述",
            "总结",
            "主要",
            "讲了什么",
        ]
        for item in replacements:
            text = text.replace(item, " ")
        parts = [part for part in re.split(r"[\s,，。？?！!、]+", text) if part]
        if not parts:
            return question.strip()
        parts.sort(key=len, reverse=True)
        return " ".join(parts[:3])

    @staticmethod
    def _keyword_parts(question: str) -> List[str]:
        core = Retriever._extract_core_phrase(question)
        return [part for part in re.split(r"\s+", core) if part]

    @staticmethod
    def _best_toc_match(nodes: List[YuqueTocNode], query: str) -> Optional[YuqueTocNode]:
        keywords = Retriever._keyword_parts(query) or [query]
        scored = []
        for node in nodes:
            score = sum(2 if keyword == node.title else 1 for keyword in keywords if keyword in node.title)
            if score > 0:
                scored.append((score, len(node.title), node))
        if not scored:
            return None
        scored.sort(key=lambda item: (-item[0], item[1]))
        return scored[0][2]

    @staticmethod
    def _subtree_from_uuid(nodes: List[YuqueTocNode], root_uuid: str) -> List[YuqueTocNode]:
        children_map: dict[str, List[YuqueTocNode]] = {}
        for node in nodes:
            children_map.setdefault(node.parent_uuid, []).append(node)
        out: List[YuqueTocNode] = []
        stack = list(reversed(children_map.get(root_uuid, [])))
        while stack and len(out) < 40:
            node = stack.pop()
            out.append(node)
            stack.extend(reversed(children_map.get(node.uuid, [])))
        return out

    @staticmethod
    def _format_toc_line(node: YuqueTocNode) -> str:
        indent = "  " * max(node.level - 1, 0)
        return f"{indent}- {node.title}"

    @staticmethod
    def _strip_html(text: str) -> str:
        no_tags = re.sub(r"<[^>]+>", " ", text)
        normalized = re.sub(r"\s+", " ", no_tags).strip()
        return normalized

    @staticmethod
    def _to_plain_text(payload: Any) -> str:
        if isinstance(payload, str):
            return Retriever._strip_html(payload)
        try:
            return Retriever._strip_html(json.dumps(payload, ensure_ascii=False, indent=2))
        except Exception:
            return str(payload)

    @staticmethod
    def _skill_requires_document_body(skill_id: Optional[str]) -> bool:
        """Skill 注入的生成任务依赖正文；不得以「只要目录」的 intent 提前返回 TOC。"""
        if not skill_id or skill_id == "stale-detector":
            return False
        return True

    async def _classify_intent(self, question: str, skill_id: Optional[str] = None) -> tuple[IntentType, str]:
        rule_intent: IntentType = "content"
        if self._wants_document_visual_content(question):
            rule_intent = "content"
        elif self._is_directory_question(question):
            rule_intent = "directory"
        elif self._is_doc_list_question(question):
            rule_intent = "doc_list"

        if not self._intent_client:
            intent, source = rule_intent, "rule"
        else:
            llm_intent = await self._classify_intent_with_llm(question)
            if llm_intent is None:
                intent, source = rule_intent, "rule_fallback"
            else:
                intent, source = llm_intent, "llm"

        if self._wants_document_visual_content(question) and intent != "content":
            logger.info("intent_doc_visual_override from=%s to=content", intent)
            intent, source = "content", f"{source}+doc_visual_override"

        if self._skill_requires_document_body(skill_id) and intent == "directory":
            return "content", f"{source}+skill_needs_body"
        return intent, source

    async def _classify_intent_with_llm(self, question: str) -> Optional[IntentType]:
        if not self._intent_client:
            return None
        prompt = (
            "你是检索路由分类器。仅输出一个标签，不要解释："
            "directory（只要目录/大纲/章节树本身，不要求读正文）、"
            "doc_list（文档列表/有哪些文档）、"
            "content（默认：包括从某文档/某节提取金句、摘要、分析、解释、对比等任何需要正文的问答）。\n"
            "若问题里出现具体知识点/小节标题并要求总结、提取、润色、分析，必须输出 content，不要输出 directory。\n"
            "若用户问某篇/某课程文档里有哪些图片、插图、截图或图片内容（要展示图而非统计全库文档），必须输出 content，不要输出 doc_list。\n"
            f"问题：{question}"
        )
        try:
            resp = await self._intent_client.chat.completions.create(
                model=settings.intent_llm_model,
                temperature=0,
                extra_body={"enable_thinking": False},
                messages=[
                    {"role": "system", "content": "只返回 directory/doc_list/content 其中一个。"},
                    {"role": "user", "content": prompt},
                ],
            )
        except Exception as exc:
            logger.warning("intent_llm_failed error=%s", exc)
            return None
        raw = (resp.choices[0].message.content or "").strip().lower()
        if "directory" in raw:
            return "directory"
        if "doc_list" in raw:
            return "doc_list"
        if "content" in raw:
            return "content"
        return None

