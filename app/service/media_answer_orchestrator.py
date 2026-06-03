from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence

from app.conversation.visitor_profile import VisitorType, detect_visitor_type
from app.conversation.lead_nudge_policy import LeadNudgePolicy
from app.core.logger import get_logger
from app.data.mcp_client import MCPDocMeta, MCPSearchResult, YuqueMCPClient
from app.db.repositories import ChatMessageRow
from app.db.repositories import ChatSessionRepository, LeadCaptureRepository
from app.rag.generator import Generator
from app.data.yuque_images import encode_image_proxy_token, is_allowed_yuque_image_url
from app.schemas.chat import ChatMediaBundle, ChatV2Response, MediaItem, SourceItem

logger = get_logger(__name__)

_IMAGE_EXT = re.compile(r"\.(png|jpg|jpeg|gif|webp|svg)(\?|$)", re.I)
_VIDEO_EXT = re.compile(r"\.(mp4|webm|mov|m3u8)(\?|$)", re.I)
_URL_RE = re.compile(r"https?://[^\s)\"'<>]+", re.I)
_MARKDOWN_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\((https?://[^)\s]+)\)", re.I)
_HTML_IMAGE_RE = re.compile(r"<img[^>]*src=[\"'](https?://[^\"']+)[\"'][^>]*>", re.I)
_HTML_VIDEO_RE = re.compile(r"<(?:video|source|iframe)[^>]*(?:src|data-src)=[\"'](https?://[^\"']+)[\"'][^>]*>", re.I)
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", re.I)
_QUERY_STOPWORDS = {
    "请",
    "介绍",
    "一下",
    "并",
    "给我",
    "相关",
    "或者",
    "平台",
    "核心",
    "功能",
    "图片",
    "视频",
}


@dataclass(frozen=True)
class _DocContext:
    doc_id: str
    title: str
    url: str
    snippet: str
    body: str


@dataclass(frozen=True)
class _MediaMatch:
    title: str
    url: str
    context: str


@dataclass(frozen=True)
class _GuideNode:
    uuid: str
    title: str
    level: int
    parent_uuid: str
    node_type: str
    url: Optional[str]
    doc_id: Optional[int]


class MediaAnswerOrchestrator:
    def __init__(
        self,
        *,
        mcp_client: YuqueMCPClient,
        generator: Generator,
        lead_policy: LeadNudgePolicy,
        lead_capture_repository: LeadCaptureRepository,
        chat_session_repository: ChatSessionRepository,
        max_images: int,
        max_videos: int,
        max_docs: int,
        prefetched_titles: Optional[Sequence[str]] = None,
        prefetched_toc_nodes: Optional[Sequence[Dict[str, Any]]] = None,
        image_rerank_mode: Literal["rule", "text_rerank"] = "text_rerank",
    ) -> None:
        self._mcp_client = mcp_client
        self._generator = generator
        self._lead_policy = lead_policy
        self._lead_capture_repository = lead_capture_repository
        self._chat_session_repository = chat_session_repository
        self._max_images = max(0, int(max_images))
        self._max_videos = max(0, int(max_videos))
        self._max_docs = max(1, int(max_docs))
        self._prefetched_titles = [t.strip() for t in (prefetched_titles or []) if (t or "").strip()]
        self._prefetched_toc_nodes = self._normalize_guide_nodes(prefetched_toc_nodes or [])
        self._image_rerank_mode = image_rerank_mode if image_rerank_mode in ("rule", "text_rerank") else "text_rerank"

    async def answer(
        self,
        *,
        question: str,
        session_id: Optional[str],
        skill_instruction: str = "",
    ) -> ChatV2Response:
        intent = self._detect_intent(question)
        sid = (session_id or "").strip()
        history = await self._chat_session_repository.list_recent_messages(session_id=sid, limit=50) if sid else []
        has_lead = await self._lead_capture_repository.has_lead_for_session(session_id=sid) if sid else False
        nudge_decision = self._lead_policy.decide(
            question=question,
            history=history,
            has_existing_lead=has_lead,
        )
        role = self._detect_role_with_history(question=question, history=history)
        docs = await self._retrieve_docs(question)
        if not docs or not self._has_reliable_doc_evidence(docs):
            guide_answer = await self._build_guidance_answer(
                question=question,
                intent=intent,
                role=role,
                nudge_text=(nudge_decision.text if nudge_decision.triggered else ""),
            )
            return ChatV2Response(
                answer=guide_answer,
                sources=[],
                fallback_used=True,
                debug={
                    "retrieval_mode": "mcp_v15",
                    "media_images": 0,
                    "media_videos": 0,
                    "intent": intent,
                    "doc_count": 0,
                    "lead_nudge_reason": nudge_decision.reason or "",
                    "guidance_triggered": True,
                    "guidance_reason": ("no_docs" if not docs else "weak_doc_evidence"),
                },
                media=ChatMediaBundle(images=[], videos=[]),
                lead_nudge_triggered=nudge_decision.triggered,
            )
        media = self._collect_media(docs, question=question, intent=intent)

        contexts = self._build_contexts(docs, media)
        sources = self._build_sources(docs)
        generation_question = self._build_generation_question(
            question=question,
            media=media,
            intent=intent,
            nudge_text=nudge_decision.text if nudge_decision.triggered else "",
            skill_instruction=skill_instruction,
        )
        answer = await self._generator.generate(
            question=generation_question,
            contexts=contexts,
            sources=sources,
            visitor_sales=True,
        )

        debug = {
            "retrieval_mode": "mcp_v15",
            "media_images": len(media.images),
            "media_videos": len(media.videos),
            "intent": intent,
            "doc_count": len(docs),
            "lead_nudge_reason": nudge_decision.reason or "",
        }
        return ChatV2Response(
            answer=answer or "暂时没有检索到可用信息，请稍后重试。",
            sources=sources,
            fallback_used=len(sources) == 0,
            debug=debug,
            media=media,
            lead_nudge_triggered=nudge_decision.triggered,
        )

    async def answer_stream(
        self,
        *,
        question: str,
        session_id: Optional[str],
        skill_instruction: str = "",
    ):
        intent = self._detect_intent(question)
        sid = (session_id or "").strip()
        history = await self._chat_session_repository.list_recent_messages(session_id=sid, limit=50) if sid else []
        has_lead = await self._lead_capture_repository.has_lead_for_session(session_id=sid) if sid else False
        nudge_decision = self._lead_policy.decide(
            question=question,
            history=history,
            has_existing_lead=has_lead,
        )
        role = self._detect_role_with_history(question=question, history=history)

        docs = await self._retrieve_docs(question)
        if not docs or not self._has_reliable_doc_evidence(docs):
            guide_answer = await self._build_guidance_answer(
                question=question,
                intent=intent,
                role=role,
                nudge_text=(nudge_decision.text if nudge_decision.triggered else ""),
            )
            response = ChatV2Response(
                answer=guide_answer,
                sources=[],
                fallback_used=True,
                debug={
                    "retrieval_mode": "mcp_v15",
                    "media_images": 0,
                    "media_videos": 0,
                    "intent": intent,
                    "doc_count": 0,
                    "lead_nudge_reason": nudge_decision.reason or "",
                    "guidance_triggered": True,
                    "guidance_reason": ("no_docs" if not docs else "weak_doc_evidence"),
                },
                media=ChatMediaBundle(images=[], videos=[]),
                lead_nudge_triggered=nudge_decision.triggered,
            )
            for token in self._chunk_tokens(response.answer):
                yield {"event": "token", "data": {"token": token}}
            yield {"event": "done", "data": response.model_dump()}
            return

        media = self._collect_media(docs, question=question, intent=intent)
        contexts = self._build_contexts(docs, media)
        sources = self._build_sources(docs)
        generation_question = self._build_generation_question(
            question=question,
            media=media,
            intent=intent,
            nudge_text=nudge_decision.text if nudge_decision.triggered else "",
            skill_instruction=skill_instruction,
        )
        answer_parts: List[str] = []
        async for token in self._generator.stream_generate(
            question=generation_question,
            contexts=contexts,
            sources=sources,
            visitor_sales=True,
        ):
            answer_parts.append(token)
            yield {"event": "token", "data": {"token": token}}
        answer = "".join(answer_parts).strip() or "暂时没有检索到可用信息，请稍后重试。"
        response = ChatV2Response(
            answer=answer,
            sources=sources,
            fallback_used=len(sources) == 0,
            debug={
                "retrieval_mode": "mcp_v15",
                "media_images": len(media.images),
                "media_videos": len(media.videos),
                "intent": intent,
                "doc_count": len(docs),
                "lead_nudge_reason": nudge_decision.reason or "",
            },
            media=media,
            lead_nudge_triggered=nudge_decision.triggered,
        )
        yield {"event": "done", "data": response.model_dump()}

    async def _build_guidance_answer(
        self,
        *,
        question: str,
        intent: str,
        role: VisitorType,
        nudge_text: str = "",
    ) -> str:
        toc_nodes = list(self._prefetched_toc_nodes)
        if toc_nodes:
            return await self._build_guidance_answer_by_toc(
                question=question,
                intent=intent,
                role=role,
                toc_nodes=toc_nodes,
                nudge_text=nudge_text,
            )

        titles = list(self._prefetched_titles)
        if not titles:
            try:
                docs = await self._mcp_client.list_docs()
            except Exception as exc:
                logger.warning("v15_mcp_list_docs_for_guide_failed err=%s", exc)
                docs = []
            titles = [d.title.strip() for d in docs if (d.title or "").strip()]
        picks = self._pick_guide_titles(titles)

        if not picks:
            picks = ["平台介绍", "使用指南", "IDEAS-PBL"]

        ask_media = "截图/视频" if intent in ("image", "video") else "图文/视频"
        examples = [
            f"请介绍《{picks[0]}》的核心内容",
            f"《{picks[1]}》里有哪些关键步骤，可否给出相关{ask_media}",
            f"《{picks[2]}》适合哪些场景",
        ]
        role_line = self._role_identity_line(role)
        base = (
            "当然可以，我来按您的场景做个快速推荐。\n\n"
            f"{role_line}\n\n"
            "我先给您几个最常咨询的方向，您选一个我就马上展开：\n"
            f"- {picks[0]}\n"
            f"- {picks[1]}\n"
            f"- {picks[2]}\n\n"
            "你可以直接复制下面任一问题继续聊，我会按对应文档给你答复：\n"
            f"1) {examples[0]}\n"
            f"2) {examples[1]}\n"
            f"3) {examples[2]}\n\n"
            "也可以直接告诉我您最想解决的问题，我会给您更贴合教学场景的建议。"
        )
        return self._append_nudge(base, nudge_text=nudge_text)

    async def _build_guidance_answer_by_toc(
        self,
        *,
        question: str,
        intent: str,
        role: VisitorType,
        toc_nodes: Sequence[_GuideNode],
        nudge_text: str,
    ) -> str:
        roots, children_map = self._build_toc_index(toc_nodes)
        role_line = self._role_identity_line(role)
        ask_media = "截图/视频" if intent in ("image", "video") else "图文/视频"
        matched = self._match_node_by_question(question=question, toc_nodes=toc_nodes)

        if not matched:
            root_titles = [n.title for n in roots[:6]]
            if not root_titles:
                root_titles = self._pick_guide_titles([n.title for n in toc_nodes])
            sample = root_titles[:3] if len(root_titles) >= 3 else (root_titles + ["平台介绍", "使用指南", "案例与社区"])[:3]
            base = (
                "当然可以，我先按您当前最关心的方向来推荐。\n\n"
                f"{role_line}\n\n"
                "您可以先从这些方向里选一个：\n"
                + "\n".join([f"- {x}" for x in root_titles])
                + "\n\n"
                "你可以直接复制：\n"
                f"1) 我想了解《{sample[0]}》\n"
                f"2) 先看《{sample[1]}》\n"
                f"3) 帮我讲讲《{sample[2]}》\n\n"
                "您选定后，我会继续推荐更细分的内容方向，并结合文档给您说明。"
            )
            return self._append_nudge(base, nudge_text=nudge_text)

        children = children_map.get(matched.uuid, [])
        if children:
            brief = await self._node_quick_brief(matched)
            child_titles = [n.title for n in children[:8]]
            sample = child_titles[:3] if len(child_titles) >= 3 else child_titles
            sample_lines = "\n".join([f"{idx + 1}) 我想看《{name}》" for idx, name in enumerate(sample)])
            body = (
                "已收到，我先给您一个简要说明：\n\n"
                + f"{brief}\n\n"
                if brief
                else "已收到，下面我给您推荐更细分的内容方向：\n\n"
            )
            base = (
                f"{role_line}\n\n"
                f"围绕《{matched.title}》，您可以继续了解：\n"
                + "\n".join([f"- {x}" for x in child_titles])
                + ("\n\n你可以直接复制：\n" + sample_lines if sample_lines else "")
                + "\n\n"
            )
            return self._append_nudge(body + base, nudge_text=nudge_text)

        base = (
            "很好，已经定位到具体目录节点。\n\n"
            f"{role_line}\n\n"
            f"当前已定位：《{matched.title}》。\n"
            f"你可以直接让我提取该文档内容，并按你的场景输出{ask_media}：\n"
            f"1) 请提取《{matched.title}》的核心内容\n"
            f"2) 《{matched.title}》里有哪些关键步骤，给我相关{ask_media}\n"
            f"3) 《{matched.title}》适合哪些人群/场景\n\n"
            "我会严格基于命中文档内容回答，给您可直接落地的建议。"
        )
        return self._append_nudge(base, nudge_text=nudge_text)

    async def _node_quick_brief(self, node: _GuideNode) -> str:
        doc_id = node.doc_id
        if doc_id is None:
            return ""
        try:
            body = await self._mcp_client.get_doc(str(doc_id))
        except Exception as exc:
            logger.warning("v15_mcp_get_doc_brief_failed doc_id=%s err=%s", doc_id, exc)
            return ""
        text = self._to_plain_text(body or "")
        if len(text) < 20:
            return ""
        return text[:120] + ("…" if len(text) > 120 else "")

    @staticmethod
    def _to_plain_text(text: str) -> str:
        t = text or ""
        t = _MARKDOWN_IMAGE_RE.sub(" ", t)
        t = _HTML_IMAGE_RE.sub(" ", t)
        t = _HTML_VIDEO_RE.sub(" ", t)
        t = _URL_RE.sub(" ", t)
        t = re.sub(r"[#>*`_\\-]{1,}", " ", t)
        t = re.sub(r"\[[^\]]+\]\([^)]+\)", " ", t)
        t = re.sub(r"<[^>]+>", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    @staticmethod
    def _chunk_tokens(text: str, size: int = 24) -> List[str]:
        if not text:
            return []
        out: List[str] = []
        for i in range(0, len(text), size):
            out.append(text[i : i + size])
        return out

    @staticmethod
    def _append_nudge(base: str, *, nudge_text: str) -> str:
        extra = (nudge_text or "").strip()
        if not extra:
            return base
        return f"{base.rstrip()}\n\n{extra}"

    @staticmethod
    def _detect_role_with_history(*, question: str, history: Sequence[ChatMessageRow]) -> VisitorType:
        current = detect_visitor_type(question)
        stable_roles = {"institution_decision_maker", "teacher", "parent", "student"}
        if current in stable_roles:
            return current
        # 回看最近用户消息，避免用户已自报身份后被后续问题覆盖成 unknown
        for row in reversed(history):
            if row.role != "user":
                continue
            vt = detect_visitor_type(row.content or "")
            if vt in stable_roles:
                return vt
        return current

    @staticmethod
    def _normalize_guide_nodes(nodes: Sequence[Dict[str, Any]]) -> List[_GuideNode]:
        out: List[_GuideNode] = []
        for item in nodes:
            title = str(item.get("title") or "").strip()
            if not title:
                continue
            raw_level = item.get("level")
            try:
                level = max(1, int(raw_level or 1))
            except Exception:
                level = 1
            raw_doc_id = item.get("doc_id")
            try:
                doc_id = int(raw_doc_id) if raw_doc_id is not None else None
            except Exception:
                doc_id = None
            out.append(
                _GuideNode(
                    uuid=str(item.get("uuid") or ""),
                    title=title,
                    level=level,
                    parent_uuid=str(item.get("parent_uuid") or ""),
                    node_type=str(item.get("node_type") or item.get("type") or ""),
                    url=(str(item.get("url") or "").strip() or None),
                    doc_id=doc_id,
                )
            )
        return out

    @staticmethod
    def _build_toc_index(toc_nodes: Sequence[_GuideNode]) -> tuple[List[_GuideNode], Dict[str, List[_GuideNode]]]:
        by_uuid: Dict[str, _GuideNode] = {n.uuid: n for n in toc_nodes if n.uuid}
        children_map: Dict[str, List[_GuideNode]] = {}
        roots: List[_GuideNode] = []
        for n in toc_nodes:
            parent_uuid = (n.parent_uuid or "").strip()
            if parent_uuid and parent_uuid in by_uuid and parent_uuid != n.uuid:
                children_map.setdefault(parent_uuid, []).append(n)
            else:
                roots.append(n)
        roots.sort(key=lambda x: (x.level, x.title))
        for key in list(children_map.keys()):
            children_map[key].sort(key=lambda x: (x.level, x.title))
        return roots, children_map

    @staticmethod
    def _match_node_by_question(question: str, toc_nodes: Sequence[_GuideNode]) -> Optional[_GuideNode]:
        q = (question or "").strip().lower()
        if not q:
            return None
        best: Optional[tuple[int, _GuideNode]] = None
        for node in toc_nodes:
            title = (node.title or "").strip().lower()
            if not title:
                continue
            score = 0
            matched = False
            if title in q:
                score += 200 + len(title)
                matched = True
            if len(q) >= 2 and q in title:
                score += 40 + len(q)
                matched = True
            for kw in MediaAnswerOrchestrator._extract_keywords(question):
                if kw and kw in title:
                    score += 6
                    matched = True
            if not matched:
                continue
            score += min(max(node.level, 1), 6)
            if best is None or score > best[0]:
                best = (score, node)
        return best[1] if best else None

    @staticmethod
    def _pick_guide_titles(titles: Sequence[str]) -> List[str]:
        preferred_keys = ("平台介绍", "使用指南", "案例")
        picked: List[str] = []
        seen: set[str] = set()
        for key in preferred_keys:
            hit = next((t for t in titles if key in t and t not in seen), None)
            if hit:
                picked.append(hit)
                seen.add(hit)
        for t in titles:
            if t in seen:
                continue
            picked.append(t)
            seen.add(t)
            if len(picked) >= 3:
                break
        return picked[:3]

    @staticmethod
    def _role_identity_line(role: VisitorType) -> str:
        if role == "institution_decision_maker":
            return "已了解您是校长/机构负责人，我会优先从落地场景、部署方式和投入产出来回答。"
        if role == "teacher":
            return "老师您好，我会优先从课堂怎么用、备课流程和作业反馈给您建议。"
        if role == "parent":
            return "家长您好，我会优先从适配年级、学习效果和使用门槛来给您建议。"
        if role == "student":
            return "同学你好，我会优先从上手路径、学习方法和练习建议来回答。"
        return "你也可以先告诉我你的身份（校长/老师/家长/学生），我会按你的场景给更精准建议。"

    async def _retrieve_docs(self, question: str) -> List[_DocContext]:
        if not self._mcp_client.enabled:
            return []
        candidates: List[MCPSearchResult] = []
        seen: set[str] = set()
        for query in self._build_search_queries_with_toc(question):
            try:
                items = await self._mcp_client.search(query)
            except Exception as exc:
                logger.warning("v15_mcp_search_failed query=%r err=%s", query, exc)
                continue
            for item in items:
                doc_id = (item.doc_id or "").strip()
                if not doc_id or doc_id in seen:
                    continue
                seen.add(doc_id)
                candidates.append(item)
                if len(candidates) >= self._max_docs:
                    break
            if len(candidates) >= self._max_docs:
                break

        async def _fetch(item: MCPSearchResult) -> _DocContext:
            body = ""
            try:
                body = await self._mcp_client.get_doc(item.doc_id)
            except Exception as exc:
                logger.warning("v15_mcp_get_doc_failed doc_id=%s err=%s", item.doc_id, exc)
            url = (item.url or "").strip()
            if not url:
                repo = (getattr(self._mcp_client, "repo_id", "") or "").strip("/")
                did = (item.doc_id or "").strip()
                if repo and did:
                    # 在 MCP 未返回 url 时提供可追溯链接
                    url = f"/{repo}/{did}"
            return _DocContext(
                doc_id=item.doc_id,
                title=item.title,
                url=url,
                snippet=item.snippet,
                body=(body or "").strip(),
            )

        docs = await asyncio.gather(*[_fetch(item) for item in candidates], return_exceptions=False)
        out = [d for d in docs if d.body or d.snippet]
        if out:
            return out

        # 搜索未命中时，退化为 list_docs 标题匹配，避免长问句直查导致 0 结果。
        try:
            docs_meta = await self._mcp_client.list_docs()
        except Exception as exc:
            logger.warning("v15_mcp_list_docs_failed err=%s", exc)
            return []
        title_candidates = self._pick_candidates_from_docs(question=question, docs=docs_meta)
        if not title_candidates:
            title_candidates = self._pick_candidates_from_toc(question=question)
        docs2 = await asyncio.gather(*[_fetch(item) for item in title_candidates], return_exceptions=False)
        return [d for d in docs2 if d.body or d.snippet]

    def _collect_media(self, docs: Sequence[_DocContext], *, question: str, intent: str) -> ChatMediaBundle:
        images: List[tuple[int, MediaItem]] = []
        videos: List[tuple[int, MediaItem]] = []
        seen_urls: set[str] = set()
        keywords = self._extract_keywords(question)
        for doc in docs:
            block = f"{doc.title}\n{doc.snippet}\n{doc.body}"
            for hit in self._extract_image_urls(block):
                u = hit.url.strip()
                if not u or u in seen_urls:
                    continue
                seen_urls.add(u)
                item = MediaItem(url=u, title=hit.title, doc_title=doc.title, doc_id=doc.doc_id)
                score = self._media_score(item=item, keywords=keywords, context=hit.context, question=question, media_kind="image", intent=intent)
                images.append((score, item))
            for hit in self._extract_video_urls(block):
                u = hit.url.strip()
                if not u or u in seen_urls:
                    continue
                seen_urls.add(u)
                item = MediaItem(url=u, title=hit.title, doc_title=doc.title, doc_id=doc.doc_id)
                score = self._media_score(item=item, keywords=keywords, context=hit.context, question=question, media_kind="video", intent=intent)
                videos.append((score, item))

        images.sort(key=lambda x: x[0], reverse=True)
        videos.sort(key=lambda x: x[0], reverse=True)

        image_items = [item for _, item in images[: self._max_images]]
        video_items = [item for _, item in videos[: self._max_videos]]
        if intent == "video":
            # 严格按访客诉求返回：要视频时仅返回视频（没有则为空）
            return ChatMediaBundle(images=[], videos=video_items)
        if intent == "image":
            # 严格按访客诉求返回：要图片时仅返回图片（没有则为空）
            return ChatMediaBundle(images=image_items, videos=[])
        # 普通问答保留可用媒体
        return ChatMediaBundle(images=image_items, videos=video_items)

    @staticmethod
    def _extract_image_urls(text: str) -> List[_MediaMatch]:
        out: List[_MediaMatch] = []
        for m in _MARKDOWN_IMAGE_RE.finditer(text or ""):
            out.append(
                _MediaMatch(
                    title=(m.group(1) or "").strip(),
                    url=(m.group(2) or "").strip(),
                    context=MediaAnswerOrchestrator._around_context(text, m.start(), m.end()),
                )
            )
        for m in _HTML_IMAGE_RE.finditer(text or ""):
            out.append(
                _MediaMatch(
                    title="",
                    url=(m.group(1) or "").strip(),
                    context=MediaAnswerOrchestrator._around_context(text, m.start(), m.end()),
                )
            )
        for m in _URL_RE.finditer(text or ""):
            u = (m.group(0) or "").strip()
            if _IMAGE_EXT.search(u):
                out.append(
                    _MediaMatch(
                        title="",
                        url=u,
                        context=MediaAnswerOrchestrator._around_context(text, m.start(), m.end()),
                    )
                )
        return out

    @staticmethod
    def _extract_video_urls(text: str) -> List[_MediaMatch]:
        out: List[_MediaMatch] = []
        for m in _HTML_VIDEO_RE.finditer(text or ""):
            out.append(
                _MediaMatch(
                    title="",
                    url=(m.group(1) or "").strip(),
                    context=MediaAnswerOrchestrator._around_context(text, m.start(), m.end()),
                )
            )
        for m in _URL_RE.finditer(text or ""):
            u = (m.group(0) or "").strip()
            if _VIDEO_EXT.search(u) or any(k in u.lower() for k in ("youku", "bilibili", "youtube", "v.qq.com")):
                out.append(
                    _MediaMatch(
                        title="",
                        url=u,
                        context=MediaAnswerOrchestrator._around_context(text, m.start(), m.end()),
                    )
                )
        for m in _MARKDOWN_LINK_RE.finditer(text or ""):
            title = (m.group(1) or "").strip()
            u = (m.group(2) or "").strip()
            title_lower = title.lower()
            if (
                _VIDEO_EXT.search(u)
                or any(k in u.lower() for k in ("youku", "bilibili", "youtube", "v.qq.com"))
                or any(k in title_lower for k in ("视频", "演示", "demo", "讲解", "录屏", "课程介绍"))
            ):
                out.append(
                    _MediaMatch(
                        title=title,
                        url=u,
                        context=MediaAnswerOrchestrator._around_context(text, m.start(), m.end()),
                    )
                )
        return out

    def _media_score(
        self,
        *,
        item: MediaItem,
        keywords: Sequence[str],
        context: str,
        question: str,
        media_kind: Literal["image", "video"],
        intent: str,
    ) -> int:
        base = 1
        text = f"{item.title} {item.doc_title}".lower()
        for kw in keywords:
            if kw and kw in text:
                base += 2
        if self._image_rerank_mode == "text_rerank":
            q_words = [w for w in self._extract_keywords(question) if w and w not in _QUERY_STOPWORDS]
            ctx = (context or "").lower()
            for w in q_words:
                if w in ctx:
                    base += 3
            if media_kind == "video" and intent == "video":
                base += 4
            if media_kind == "image" and intent == "image":
                base += 4
        return base

    @staticmethod
    def _around_context(text: str, start: int, end: int, window: int = 140) -> str:
        if not text:
            return ""
        left = max(0, start - window)
        right = min(len(text), end + window)
        chunk = text[left:right]
        chunk = _URL_RE.sub(" ", chunk)
        return re.sub(r"\s+", " ", chunk).strip()

    @staticmethod
    def _detect_intent(question: str) -> str:
        q = (question or "").lower()
        if any(k in q for k in ("视频", "演示", "教程", "讲解", "操作")):
            return "video"
        if any(k in q for k in ("图片", "截图", "流程图", "示意图", "界面图", "步骤图")):
            return "image"
        return "text"

    @staticmethod
    def _extract_keywords(question: str) -> List[str]:
        parts = re.split(r"[\s,，。！？?、]+", (question or "").lower())
        return [p for p in parts if len(p) >= 2][:8]

    @staticmethod
    def _build_search_queries(question: str) -> List[str]:
        q = (question or "").strip()
        if not q:
            return []
        compact = re.sub(r"[？?。！，、：:；;（）()]", " ", q).strip()
        parts = [x.strip() for x in re.split(r"\s+", compact.lower()) if x.strip()]
        keywords = [p for p in parts if len(p) >= 2 and p not in _QUERY_STOPWORDS][:6]
        core = " ".join(keywords[:3]).strip()
        pieces = [compact, core, q]
        out: List[str] = []
        for p in pieces:
            if p and p not in out:
                out.append(p)
        return out

    def _build_search_queries_with_toc(self, question: str) -> List[str]:
        out = self._build_search_queries(question)
        toc_match = self._match_node_by_question(question=question, toc_nodes=self._prefetched_toc_nodes)
        if not toc_match:
            return out
        candidates = [toc_match.title]
        candidates.extend([n.title for n in self._children_of_node(toc_match.uuid)[:2]])
        for p in candidates:
            if p and p not in out:
                out.append(p)
        return out

    def _pick_candidates_from_docs(self, *, question: str, docs: Sequence[MCPDocMeta]) -> List[MCPSearchResult]:
        q = (question or "").lower()
        kws = [k for k in self._extract_keywords(question) if k not in _QUERY_STOPWORDS]
        scored: List[tuple[int, MCPSearchResult]] = []
        for doc in docs:
            title = (doc.title or "").strip()
            if not title:
                continue
            title_l = title.lower()
            score = 0
            if title_l and title_l in q:
                score += 100
            if len(q) >= 2 and q in title_l:
                score += 20
            score += self._title_keyword_score(title_l=title_l, keywords=kws)
            score += self._char_overlap_score(q=q, title=title_l)
            doc_id = str(getattr(doc, "doc_id", "") or "")
            if score > 0 and doc_id:
                scored.append(
                    (
                        score,
                        MCPSearchResult(
                            doc_id=doc_id,
                            title=title,
                            url=str(getattr(doc, "url", "") or ""),
                            snippet=title,
                        ),
                    )
                )
        scored.sort(key=lambda x: x[0], reverse=True)
        picked = [x for _, x in scored[: self._max_docs]]
        return picked

    def _pick_candidates_from_toc(self, *, question: str) -> List[MCPSearchResult]:
        matched = self._match_node_by_question(question=question, toc_nodes=self._prefetched_toc_nodes)
        if not matched:
            return []
        nodes = [matched, *self._children_of_node(matched.uuid)[:5]]
        out: List[MCPSearchResult] = []
        seen: set[str] = set()
        for node in nodes:
            if node.doc_id is None:
                continue
            doc_id = str(node.doc_id)
            if not doc_id or doc_id in seen:
                continue
            seen.add(doc_id)
            out.append(
                MCPSearchResult(
                    doc_id=doc_id,
                    title=node.title,
                    url=(node.url or ""),
                    snippet=node.title,
                )
            )
            if len(out) >= self._max_docs:
                break
        return out

    def _children_of_node(self, node_uuid: str) -> List[_GuideNode]:
        if not node_uuid:
            return []
        return [n for n in self._prefetched_toc_nodes if (n.parent_uuid or "") == node_uuid]

    @staticmethod
    def _title_keyword_score(*, title_l: str, keywords: Sequence[str]) -> int:
        score = 0
        for k in keywords:
            if k and k in title_l:
                score += 6
        return score

    @staticmethod
    def _char_overlap_score(*, q: str, title: str) -> int:
        q_set = {c for c in q if c.strip()}
        t_set = {c for c in title if c.strip()}
        if not q_set or not t_set:
            return 0
        overlap = len(q_set.intersection(t_set))
        return min(overlap, 8)

    @staticmethod
    def _build_sources(docs: Sequence[_DocContext]) -> List[SourceItem]:
        return [
            SourceItem(
                title=doc.title or "语雀文档",
                url=MediaAnswerOrchestrator._normalize_doc_url(doc.url),
                source_type="mcp",
                snippet=(doc.snippet or doc.body or "")[:200],
                doc_id=doc.doc_id or None,
            )
            for doc in docs
        ]

    @staticmethod
    def _normalize_doc_url(url: str) -> str:
        u = (url or "").strip()
        if not u:
            return ""
        if u.startswith("http://") or u.startswith("https://"):
            return u
        if u.startswith("/"):
            return f"https://www.yuque.com{u}"
        return f"https://www.yuque.com/{u.lstrip('/')}"

    @staticmethod
    def _build_contexts(docs: Sequence[_DocContext], media: ChatMediaBundle) -> List[str]:
        contexts: List[str] = []
        for doc in docs[:6]:
            body = (doc.body or doc.snippet or "").strip()
            if not body:
                continue
            contexts.append(f"文档标题：{doc.title}\n{body[:2800]}")
        if media.videos:
            lines = [f"- 参考视频{i + 1}（{x.title or x.doc_title or '视频'}）：{x.url}" for i, x in enumerate(media.videos)]
            contexts.append("已检索到相关视频：\n" + "\n".join(lines))
        if media.images:
            lines = [f"- 参考图{i + 1}（{x.title or x.doc_title or '图片'}）：{x.url}" for i, x in enumerate(media.images)]
            contexts.append("已检索到相关图片：\n" + "\n".join(lines))
        return contexts or ["未检索到可靠上下文，请给出简洁且诚实的说明。"]

    @staticmethod
    def _build_generation_question(
        *,
        question: str,
        media: ChatMediaBundle,
        intent: str,
        nudge_text: str,
        skill_instruction: str,
    ) -> str:
        if media.images or media.videos:
            if intent == "video":
                media_hint = (
                    f"当前可用视频数量={len(media.videos)}，图片数量={len(media.images)}。"
                    "用户偏好视频，优先围绕视频内容回答；仅在必要时补充图片。"
                )
            elif intent == "image":
                media_hint = (
                    f"当前可用图片数量={len(media.images)}，视频数量={len(media.videos)}。"
                    "用户偏好图片，优先围绕图片内容回答；仅在必要时补充视频。"
                )
            else:
                media_hint = (
                    f"当前可用图片数量={len(media.images)}，视频数量={len(media.videos)}。"
                    "结合文本与媒体给出简洁回答。若素材有帮助，请在文字里自然引用“参考图1/参考图2/参考视频1”，"
                    "说明这些素材分别展示了什么，不要只丢图片不解释。"
                )
        else:
            media_hint = (
                "当前未命中可直接返回的图片/视频素材。"
                "请直接回答问题，不要主动强调“没有图片/视频”，除非用户明确追问素材可得性。"
            )
        nudge_hint = f"\n如自然不突兀，可在结尾补一句：{nudge_text}" if nudge_text else ""
        skill_hint = f"\n补充策略：{skill_instruction}" if skill_instruction else ""
        return (
            "请以销售顾问口吻回答，控制在约100-150字，避免长篇。"
            "结构固定为：先1句自然承接；再1句总领；再用2-4条 `- **关键词**：说明` 讲重点；最后1句自然互动提问。"
            "如果展示图片/视频，先写1句引导语，再简短说明参考图或参考视频展示了什么。"
            "如果上下文不足，需明确说明。"
            "禁止编造目录中不存在的模块、功能、案例或指标；"
            "回答必须可在给定上下文中找到依据。"
            f"\n{media_hint}{skill_hint}{nudge_hint}\n\n用户问题：{question}"
        )

    @staticmethod
    def _has_reliable_doc_evidence(docs: Sequence[_DocContext]) -> bool:
        # 至少一篇文档正文达到基础长度，才允许进入“生成回答”链路
        for d in docs:
            body_len = len((d.body or "").strip())
            if body_len >= 80:
                return True
        return False


def doc_body_has_images(doc: _DocContext) -> bool:
    block = f"{doc.title}\n{doc.snippet}\n{doc.body}"
    return bool(MediaAnswerOrchestrator._extract_image_urls(block))


def collect_media_from_doc_contexts(
    docs: Sequence[_DocContext],
    *,
    question: str,
    max_images: int,
    max_videos: int,
    primary_doc_title: str = "",
) -> ChatMediaBundle:
    """从已拉取的文档正文中抽取图片/视频；若指定 primary_doc_title 则仅从该篇取图，避免跨文档混图。"""
    scoped = list(docs)
    pt = (primary_doc_title or "").strip()
    if pt and scoped:
        scoped = [d for d in scoped if (d.title or "").strip() == pt] or [scoped[0]]
    if not scoped:
        return ChatMediaBundle()
    images: List[tuple[int, MediaItem]] = []
    videos: List[tuple[int, MediaItem]] = []
    seen_urls: set[str] = set()
    keywords = MediaAnswerOrchestrator._extract_keywords(question)
    for doc in scoped:
        block = f"{doc.title}\n{doc.snippet}\n{doc.body}"
        for hit in MediaAnswerOrchestrator._extract_image_urls(block):
            u = hit.url.strip()
            if not u or u in seen_urls:
                continue
            seen_urls.add(u)
            item = MediaItem(url=u, title=hit.title, doc_title=doc.title, doc_id=doc.doc_id)
            score = 1
            ctx = (hit.context or "").lower()
            for kw in keywords:
                if kw and kw in ctx:
                    score += 3
            images.append((score, item))
        for hit in MediaAnswerOrchestrator._extract_video_urls(block):
            u = hit.url.strip()
            if not u or u in seen_urls:
                continue
            seen_urls.add(u)
            item = MediaItem(url=u, title=hit.title, doc_title=doc.title, doc_id=doc.doc_id)
            score = 1
            videos.append((score, item))
    images.sort(key=lambda x: x[0], reverse=True)
    videos.sort(key=lambda x: x[0], reverse=True)
    return apply_yuque_proxy_to_media(
        ChatMediaBundle(
            images=[item for _, item in images[: max(0, int(max_images))]],
            videos=[item for _, item in videos[: max(0, int(max_videos))]],
        )
    )


def apply_yuque_proxy_to_media(media: ChatMediaBundle) -> ChatMediaBundle:
    """语雀媒体 URL 统一走同源代理，否则浏览器图片/视频常因鉴权或 CORS 无法直接访问。"""
    out_images: List[MediaItem] = []
    for item in media.images:
        u = (item.url or "").strip()
        if u.startswith("/yuque/asset"):
            out_images.append(item)
            continue
        if is_allowed_yuque_image_url(u):
            u = f"/yuque/asset?t={encode_image_proxy_token(u)}"
        out_images.append(
            MediaItem(
                url=u,
                title=item.title,
                doc_title=item.doc_title,
                doc_id=item.doc_id,
                summary=item.summary,
            )
        )
    out_videos: List[MediaItem] = []
    for item in media.videos:
        u = (item.url or "").strip()
        if u.startswith("/yuque/asset"):
            out_videos.append(item)
            continue
        if is_allowed_yuque_image_url(u):
            u = f"/yuque/asset?t={encode_image_proxy_token(u)}"
        out_videos.append(
            MediaItem(
                url=u,
                title=item.title,
                doc_title=item.doc_title,
                doc_id=item.doc_id,
                summary=item.summary,
            )
        )
    return ChatMediaBundle(images=out_images, videos=out_videos)
