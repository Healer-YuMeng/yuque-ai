from __future__ import annotations

import asyncio
import re
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, Tuple

from app.conversation.catalog_state_machine import (
    CatalogDialogState,
    CatalogStateMachine,

)
from app.conversation.chat_display import display_name_for_chat
from app.conversation.persona_template import build_model_agnostic_sales_persona_template
from app.conversation.profile_extractor import ProfileExtractor
from app.conversation.skill_planner import format_skill_instructions_block, plan_sales_skills
from app.conversation.turn_trace import SkillTraceItem, TurnTraceBuilder, empty_guide_trace
from app.conversation.toc_catalog import CatalogNode, TocCatalogIndex, _title_match_score
from app.conversation.v4_lead_outreach import V4LeadOutreach
from app.core.config import settings
from app.core.logger import get_logger
from app.data.mcp_client import MCPSearchResult, YuqueMCPClient
from app.db.profile_repository import ChatSessionProfile, ChatSessionProfileRepository
from app.db.repositories import ChatMessageRow
from app.rag.generator import Generator
from app.schemas.chat import ChatMediaBundle, ChatV4Response, SourceItem
from app.service.media_answer_orchestrator import (
    _DocContext,
    apply_yuque_proxy_to_media,
    collect_media_from_doc_contexts,
    doc_body_has_images,
)
from app.service.v4_vision_enrichment import enrich_media_bundle_with_vision

logger = get_logger(__name__)
_SESSION_META_KEY = "_session"
_LEAD_META_KEY = "_lead"
_DOC_CONTEXT_CACHE_TTL_S = 15 * 60
_NODE_RETRIEVAL_CACHE_TTL_S = 10 * 60
_FIELD_LABELS = {
    "name": "称呼",
    "org_name": "单位",
    "contact": "联系方式",
    "interested_product": "感兴趣产品",
}


class SalesDialogOrchestratorV4:
    """V4：语雀目录状态机 + 目录内关联检索 + 图文回答（旁路，不改 V1/V2/V3）。"""

    def __init__(
        self,
        *,
        mcp_client: YuqueMCPClient,
        generator: Generator,
        profile_repo: ChatSessionProfileRepository,
        toc_nodes: Sequence[Dict[str, Any]],
        lead_outreach: V4LeadOutreach,
    ) -> None:
        self._mcp_client = mcp_client
        self._generator = generator
        self._profile_repo = profile_repo
        self._extractor = ProfileExtractor()
        self._lead_outreach = lead_outreach
        self._catalog = TocCatalogIndex(toc_nodes)
        self._fsm = CatalogStateMachine(self._catalog)
        self._doc_context_cache: Dict[str, Tuple[float, _DocContext]] = {}
        self._node_retrieval_cache: Dict[Tuple[str, str], Tuple[float, List[_DocContext], List[SourceItem]]] = {}

    async def answer_stream(
        self,
        *,
        question: str,
        session_id: str,
        history: Sequence[ChatMessageRow],
        selected_doc_ids: Sequence[int] = (),
    ) -> AsyncIterator[dict[str, Any]]:
        sid = (session_id or "").strip()
        profile = await self._profile_repo.get_profile(session_id=sid) if sid else None
        catalog_state = (
            await self._profile_repo.get_catalog_state(session_id=sid) if sid else CatalogDialogState()
        )

        update = await self._extractor.extract_update(question=question, history=history, current_profile=profile)
        if sid:
            merged_interests = None
            if update.interests is not None:
                base = dict(profile.interests) if profile else {}
                base.update(update.interests)
                merged_interests = base
            # 身份/姓名/单位：只有新值非空时才覆写，防止后续轮次空串清掉已有画像
            existing_vt = (profile.visitor_type if profile else "") or ""
            new_vt_raw = str(update.visitor_type or "").strip()
            # 禁止把已知身份降级为空或通用值
            write_vt = new_vt_raw if new_vt_raw and (not existing_vt or new_vt_raw != existing_vt) else None

            existing_name = (profile.display_name if profile else "") or ""
            new_name_raw = (update.display_name or "").strip()
            # 新名字比旧名字更长（更完整）时才覆写
            write_name = new_name_raw if new_name_raw and len(new_name_raw) >= len(existing_name) else None

            existing_org = (profile.org_name if profile else "") or ""
            new_org_raw = (update.org_name or "").strip()
            write_org = new_org_raw if new_org_raw else (None if existing_org else None)

            await self._profile_repo.upsert_profile(
                session_id=sid,
                display_name=write_name,
                visitor_type=write_vt,
                org_name=write_org if write_org else None,
                interests=merged_interests,
            )
            profile = await self._profile_repo.get_profile(session_id=sid)

        forced_anchor: Optional[CatalogNode] = None
        for did in selected_doc_ids:
            node = self._catalog.find_by_doc_id(int(did))
            if node is not None:
                forced_anchor = node
                break

        if forced_anchor is not None:
            level = self._catalog.dialog_level(forced_anchor)
            new_state = CatalogDialogState(
                node_uuid=forced_anchor.uuid,
                path_titles=list(forced_anchor.path_titles),
                root_guide_shown=level > 0,
                dialog_level=level,
            )
            anchor, action = forced_anchor, "anchor"
        else:
            new_state, anchor, action = self._fsm.apply_user_turn(question=question, state=catalog_state)
        if action == "reset":
            new_state = CatalogDialogState()
            anchor = None

        if sid:
            await self._upsert_session_meta(
                session_id=sid,
                profile=profile,
                scene_picked=bool(anchor is not None or selected_doc_ids),
                module_scope=_infer_module_scope(anchor),
                catalog_anchor=new_state.path_titles,
            )
            profile = await self._profile_repo.get_profile(session_id=sid)

        repeat_feedback_field = _detect_repeat_feedback_field(question=question, history=history)
        if sid and repeat_feedback_field:
            await self._upsert_session_meta(
                session_id=sid,
                profile=profile,
                scene_picked=False,
                module_scope="",
                catalog_anchor=(),
                add_suppressed_fields=[repeat_feedback_field],
            )
            profile = await self._profile_repo.get_profile(session_id=sid)

        if sid:
            await self._profile_repo.save_catalog_state(session_id=sid, state=new_state)

        path_for_lead = list(new_state.path_titles) or ([anchor.title] if anchor else [])
        lead_turn = await self._lead_outreach.ingest_user_turn(
            session_id=sid,
            question=question,
            profile=profile,
            catalog_path=path_for_lead,
        )
        if sid and lead_turn.interests_patch:
            base = dict(profile.interests) if profile and profile.interests else {}
            base.update(lead_turn.interests_patch)
            await self._profile_repo.upsert_profile(session_id=sid, interests=base)
            profile = await self._profile_repo.get_profile(session_id=sid)

        # 深度内容 / 已锚定节点：检索主文档 + 目录内关联文档后讲解
        if anchor and (new_state.dialog_level >= 2 or action == "anchor"):
            async for event in self._answer_at_node(
                question=question,
                profile=profile,
                session_id=sid,
                node=anchor,
                state=new_state,
                history=history,
                lead_contact_detected=lead_turn.contact_detected,
                repeat_feedback_field=repeat_feedback_field,
            ):
                yield event
            return

        # 状态 0/1：目录内自然引导（仅当前层级子节点，不跳回根三大项）
        if self._fsm.should_show_root_guide(new_state) or new_state.dialog_level <= 1:
            candidates = self._fsm.guide_candidates(new_state, anchor)
            if candidates:
                msg = _build_natural_guide(profile=profile, state=new_state, candidates=candidates, history=history)
                if repeat_feedback_field:
                    msg = _build_repeat_feedback_apology(repeat_feedback_field) + "\n\n" + msg
                if sid and new_state.dialog_level <= 0:
                    new_state.root_guide_shown = True
                    await self._profile_repo.save_catalog_state(session_id=sid, state=new_state)
                for ch in msg:
                    yield {"event": "token", "data": {"token": ch}}
                guide_dbg = empty_guide_trace(
                    pipeline="v4_guide",
                    catalog_path=new_state.path_titles,
                    dialog_level=new_state.dialog_level,
                )
                guide_dbg.update(
                    {
                        "mode": "v4_guide",
                        "dialog_level": new_state.dialog_level,
                        "path": new_state.path_titles,
                        "contact_detected": lead_turn.contact_detected,
                        "module_scope": _read_session_meta(profile).get("module_scope", ""),
                    }
                )
                yield {
                    "event": "done",
                    "data": ChatV4Response(
                        answer=msg,
                        sources=[],
                        fallback_used=True,
                        debug=guide_dbg,
                        media=ChatMediaBundle(),
                        lead_nudge_triggered=False,
                        trial_apply_available=bool(
                            (profile.interests or {}).get("_lead", {}).get("wants_trial")
                            if profile and isinstance(profile.interests, dict)
                            else False
                        ),
                    ).model_dump(),
                }
                return

        fallback = _build_soft_clarify(profile=profile, state=new_state)
        if repeat_feedback_field:
            fallback = _build_repeat_feedback_apology(repeat_feedback_field) + "\n\n" + fallback
        for ch in fallback:
            yield {"event": "token", "data": {"token": ch}}
        clarify_dbg = empty_guide_trace(
            pipeline="v4_clarify",
            catalog_path=new_state.path_titles,
            dialog_level=new_state.dialog_level,
        )
        clarify_dbg.update({"mode": "v4_clarify", "path": new_state.path_titles})
        yield {
            "event": "done",
            "data": ChatV4Response(
                answer=fallback,
                sources=[],
                fallback_used=True,
                debug=clarify_dbg,
                media=ChatMediaBundle(),
                lead_nudge_triggered=False,
                trial_apply_available=False,
            ).model_dump(),
        }

    async def _answer_at_node(
        self,
        *,
        question: str,
        profile: Optional[ChatSessionProfile],
        session_id: str,
        node: CatalogNode,
        state: CatalogDialogState,
        history: Sequence[ChatMessageRow],
        lead_contact_detected: bool = False,
        repeat_feedback_field: str = "",
    ) -> AsyncIterator[dict[str, Any]]:
        yield {"event": "stage", "data": {"stage": "retrieving", "detail": "正在整理相关内容…", "mode": "v4"}}
        related_nodes = self._catalog.related_in_catalog(node, limit=3)
        siblings = self._catalog.siblings_of(node, include_self=False)[:2]
        subtree_nodes = self._catalog.descendants_of(node, include_self=True)[:4]
        scope_nodes: List[CatalogNode] = []
        seen_scope: set[str] = set()
        for cur in subtree_nodes + siblings:
            key = cur.uuid or cur.title
            if not key or key in seen_scope:
                continue
            seen_scope.add(key)
            scope_nodes.append(cur)
        trace = TurnTraceBuilder(
            pipeline="v4_content",
            catalog_path=node.path_titles,
            dialog_level=state.dialog_level,
        )
        sid = (session_id or "").strip()
        path_label = " / ".join(node.path_titles)
        follow_up_fast_path = _looks_like_progressive_follow_up(question=question, history=history)
        doc_ctx, sources, retrieval_dbg = await self._retrieve_for_nodes(
            scope_nodes,
            catalog_path=node.path_titles,
            trace=trace,
            primary_title=node.title,
            session_id=sid,
            cache_scope=node.uuid or path_label or node.title,
            prefer_cached=follow_up_fast_path,
        )
        primary_doc = _pick_primary_doc(doc_ctx, anchor_title=node.title)
        show_media = _should_show_media(question=question, node=node, primary=primary_doc, docs=doc_ctx)
        media = ChatMediaBundle()
        media_docs: List[_DocContext] = []
        vision_dbg: Dict[str, Any] = {}
        candidate_media = ChatMediaBundle()
        vision_task: Optional[asyncio.Task[Tuple[ChatMediaBundle, List[str], Dict[str, Any]]]] = None
        if show_media and doc_ctx:
            media_docs = _media_scope_docs(question=question, primary=primary_doc, docs=doc_ctx)
            candidate_media = collect_media_from_doc_contexts(
                media_docs,
                question=question,
                max_images=min(6, max(4, settings.chat_v15_max_images + 1)),
                max_videos=min(3, max(2, settings.chat_v15_max_videos + 1)),
                primary_doc_title="" if len(media_docs) > 1 else node.title,
            )
            show_media = bool(candidate_media.images or candidate_media.videos)
            if show_media:
                media = apply_yuque_proxy_to_media(
                    ChatMediaBundle(
                        images=list(candidate_media.images[: max(3, settings.chat_v15_max_images)]),
                        videos=list(candidate_media.videos[: max(0, settings.chat_v15_max_videos)]),
                    )
                )
                vision_task = asyncio.create_task(
                    enrich_media_bundle_with_vision(
                        candidate_media,
                        question=question,
                        max_images=max(3, settings.chat_v15_max_images),
                        max_videos=settings.chat_v15_max_videos,
                    )
                )
        if sid and doc_ctx:
            await self._profile_repo.touch_focus_docs(session_id=sid, doc_ids=[d.doc_id for d in doc_ctx if d.doc_id])
            profile = await self._profile_repo.get_profile(session_id=sid)

        planned_skills = plan_sales_skills(
            question,
            catalog_path=node.path_titles,
            dialog_level=state.dialog_level,
        )
        trace.set_skills(
            [SkillTraceItem(skill_id=s.skill_id, reason=s.reason) for s in planned_skills]
        )
        skill_block = format_skill_instructions_block(planned_skills)
        prompt = _build_v4_prompt(
            question=question,
            profile=profile,
            catalog_path=path_label,
            dialog_level=state.dialog_level,
            related_titles=[n.title for n in related_nodes if n.title != node.title],
            has_media=bool(media.images or media.videos),
            history=history,
            skill_instructions=skill_block,
        )
        primary_title = node.title
        contexts = [
            _build_tagged_context(d, is_primary=(d.title == primary_title))
            for d in doc_ctx
            if d.body or d.snippet
        ]
        yield {"event": "stage", "data": {"stage": "generating", "detail": "正在组织讲解…", "mode": "v4"}}
        if repeat_feedback_field:
            apology = _build_repeat_feedback_apology(repeat_feedback_field) + "\n\n"
            for ch in apology:
                yield {"event": "token", "data": {"token": ch}}
        async for token in self._generator.stream_generate(
            question=prompt,
            contexts=contexts or ["（当前节点暂无可用正文，请用一句话说明你想了解的场景。）"],
            sources=sources,
            visitor_sales=True,
        ):
            yield {"event": "token", "data": {"token": token}}

        tail = ""
        # 仅在用户明确发出"想看别的"信号时才追加关联推荐，且只推同目录节点
        if state.dialog_level < 3 and related_nodes and _user_wants_more(question):
            same_module = [
                n for n in related_nodes if n.title and n.title != node.title
            ][:2]
            if same_module:
                names_str = "、".join(n.title for n in same_module)
                tail = f"\n\n案例库里还有 {names_str} 的落地内容，您也可以了解一下。"
        lead_end = await self._lead_outreach.evaluate_end_of_turn(
            session_id=sid,
            question=question,
            history=history,
            profile=profile,
            dialog_level=state.dialog_level,
            is_guide_only=False,
        )
        if sid and lead_end.asked_field:
            await self._upsert_session_meta(
                session_id=sid,
                profile=profile,
                scene_picked=False,
                module_scope="",
                catalog_anchor=(),
                add_asked_fields=[lead_end.asked_field],
            )
            profile = await self._profile_repo.get_profile(session_id=sid)
        tail += lead_end.append_text
        if tail:
            for ch in tail:
                yield {"event": "token", "data": {"token": ch}}

        if vision_task is not None:
            try:
                enriched_media, _, vision_dbg = await vision_task
                if enriched_media.images or enriched_media.videos:
                    media = apply_yuque_proxy_to_media(enriched_media)
            except Exception:
                logger.exception("v4_vision_enrichment_failed")
                vision_dbg = {"vision_media_skipped": "error"}

        dbg: Dict[str, Any] = {
            "mode": "v4_content",
            "dialog_level": state.dialog_level,
            "path": node.path_titles,
            "related": [n.title for n in related_nodes],
            "retrieval_scope": "subtree_plus_siblings",
            "lead_nudge_reason": lead_end.nudge.reason or "",
            "contact_detected": lead_contact_detected,
        }
        dbg.update(retrieval_dbg)
        dbg.update(vision_dbg)
        dbg = trace.attach_debug(dbg)
        yield {
            "event": "done",
            "data": ChatV4Response(
                answer="",
                sources=sources,
                fallback_used=not bool(sources),
                debug=dbg,
                media=media,
                lead_nudge_triggered=lead_end.nudge.triggered,
                trial_apply_available=lead_end.trial_apply_available,
            ).model_dump(),
        }

    async def _retrieve_for_nodes(
        self,
        nodes: Sequence[CatalogNode],
        *,
        catalog_path: Sequence[str],
        trace: TurnTraceBuilder | None = None,
        primary_title: str = "",
        session_id: str = "",
        cache_scope: str = "",
        prefer_cached: bool = False,
    ) -> tuple[List[_DocContext], List[SourceItem], Dict[str, Any]]:
        cache_key = self._node_cache_key(session_id=session_id, cache_scope=cache_scope or "default")
        cached_payload = self._get_cached_node_payload(cache_key) if cache_key else None
        if prefer_cached and cached_payload is not None:
            doc_ctx, sources = cached_payload
            if trace is not None:
                self._record_cached_documents(trace=trace, docs=doc_ctx, primary_title=primary_title)
            return doc_ctx, sources, {
                "retrieval_cache_hit": True,
                "retrieval_skipped_search": True,
                "retrieval_cache_scope": cache_scope,
            }

        hits: List[MCPSearchResult] = []
        seen: set[str] = set()
        path_prefix = " / ".join(catalog_path)
        direct_doc_hits = 0
        search_skipped = False
        doc_cache_hits = 0

        search_queries: List[Tuple[str, str]] = []
        for node in nodes:
            if node.doc_id is not None:
                did = str(node.doc_id)
                if did not in seen:
                    seen.add(did)
                    direct_doc_hits += 1
                    hits.append(
                        MCPSearchResult(
                            doc_id=did,
                            title=node.title,
                            url=(node.url or ""),
                            snippet="",
                        )
                    )
            if _should_skip_mcp_search(
                prefer_cached=prefer_cached,
                direct_doc_hits=direct_doc_hits,
                hit_count=len(hits),
            ):
                search_skipped = True
                continue
            query = f"{path_prefix} {node.title}".strip()
            if len(search_queries) < 2:
                search_queries.append((query, node.title))

        async def _search(query: str) -> List[MCPSearchResult]:
            try:
                return await self._mcp_client.search(query)
            except Exception as exc:
                logger.warning("v4_mcp_search_failed query=%r err=%s", query, exc)
                return []

        if search_queries:
            searched_batches = await asyncio.gather(*[_search(query) for query, _ in search_queries], return_exceptions=False)
            for idx, searched in enumerate(searched_batches):
                query = search_queries[idx][0]
                if trace is not None:
                    trace.record_search(query=query, hit_count=len(searched))
                for h in searched:
                    did = (h.doc_id or "").strip()
                    if not did or did in seen:
                        continue
                    seen.add(did)
                    hits.append(h)
                    if len(hits) >= settings.chat_v15_max_docs:
                        break
                if len(hits) >= settings.chat_v15_max_docs:
                    break

        async def _fetch(item: MCPSearchResult) -> _DocContext:
            nonlocal doc_cache_hits
            cached_doc = self._get_cached_doc_context(item.doc_id)
            if cached_doc is not None:
                doc_cache_hits += 1
                return _DocContext(
                    doc_id=cached_doc.doc_id,
                    title=item.title or cached_doc.title,
                    url=(item.url or cached_doc.url or "").strip(),
                    snippet=(item.snippet or cached_doc.snippet or "")[:400],
                    body=cached_doc.body,
                )
            body = ""
            try:
                body = await self._mcp_client.get_doc(item.doc_id)
            except Exception as exc:
                logger.warning("v4_get_doc_failed doc_id=%s err=%s", item.doc_id, exc)
            if trace is not None:
                trace.record_get_doc(
                    doc_id=item.doc_id,
                    title=item.title or "",
                    body_chars=len((body or "").strip()),
                )
            doc = _DocContext(
                doc_id=item.doc_id,
                title=item.title or "",
                url=(item.url or "").strip(),
                snippet=(item.snippet or "")[:400],
                body=(body or "").strip(),
            )
            if doc.doc_id and doc.body:
                self._set_cached_doc_context(doc.doc_id, doc)
            return doc

        doc_ctx = await asyncio.gather(*[_fetch(h) for h in hits[: settings.chat_v15_max_docs]], return_exceptions=False)
        doc_ctx = [d for d in doc_ctx if d.body or d.snippet]
        sources: List[SourceItem] = []
        anchor = (primary_title or "").strip()
        for d in doc_ctx:
            role = "primary" if anchor and (d.title or "").strip() == anchor else "related"
            if trace is not None:
                trace.add_document(
                    doc_id=d.doc_id or "",
                    title=d.title or "",
                    role=role,
                    snippet=(d.snippet or d.body or "")[:200],
                    source_type="mcp",
                )
            sources.append(
                SourceItem(
                    title=d.title,
                    url=(d.url or None),
                    source_type="mcp",
                    snippet=(d.snippet or d.body or "")[:200] or None,
                    doc_id=d.doc_id or None,
                )
            )
        if cache_key and doc_ctx:
            self._set_cached_node_payload(cache_key, doc_ctx, sources)
        return doc_ctx, sources, {
            "retrieval_cache_hit": False,
            "retrieval_cache_scope": cache_scope,
            "retrieval_direct_doc_hits": direct_doc_hits,
            "retrieval_doc_cache_hits": doc_cache_hits,
            "retrieval_skipped_search": search_skipped,
        }

    @staticmethod
    def _node_cache_key(*, session_id: str, cache_scope: str) -> Tuple[str, str] | None:
        sid = (session_id or "").strip()
        scope = (cache_scope or "").strip()
        if not sid or not scope:
            return None
        return sid, scope

    @staticmethod
    def _cache_fresh(ts: float, ttl_s: int) -> bool:
        return (time.time() - ts) <= max(1, int(ttl_s))

    def _get_cached_doc_context(self, doc_id: str) -> Optional[_DocContext]:
        did = (doc_id or "").strip()
        if not did:
            return None
        payload = self._doc_context_cache.get(did)
        if not payload:
            return None
        ts, doc = payload
        if not self._cache_fresh(ts, _DOC_CONTEXT_CACHE_TTL_S):
            self._doc_context_cache.pop(did, None)
            return None
        return doc

    def _set_cached_doc_context(self, doc_id: str, doc: _DocContext) -> None:
        did = (doc_id or "").strip()
        if not did:
            return
        self._doc_context_cache[did] = (time.time(), doc)

    def _get_cached_node_payload(
        self,
        cache_key: Tuple[str, str],
    ) -> Optional[Tuple[List[_DocContext], List[SourceItem]]]:
        payload = self._node_retrieval_cache.get(cache_key)
        if not payload:
            return None
        ts, doc_ctx, sources = payload
        if not self._cache_fresh(ts, _NODE_RETRIEVAL_CACHE_TTL_S):
            self._node_retrieval_cache.pop(cache_key, None)
            return None
        return list(doc_ctx), list(sources)

    def _set_cached_node_payload(
        self,
        cache_key: Tuple[str, str],
        doc_ctx: Sequence[_DocContext],
        sources: Sequence[SourceItem],
    ) -> None:
        self._node_retrieval_cache[cache_key] = (time.time(), list(doc_ctx), list(sources))

    @staticmethod
    def _record_cached_documents(
        *,
        trace: TurnTraceBuilder,
        docs: Sequence[_DocContext],
        primary_title: str,
    ) -> None:
        anchor = (primary_title or "").strip()
        for d in docs:
            role = "primary" if anchor and (d.title or "").strip() == anchor else "related"
            trace.add_document(
                doc_id=d.doc_id or "",
                title=d.title or "",
                role=role,
                snippet=(d.snippet or d.body or "")[:200],
                source_type="mcp",
            )

    async def _upsert_session_meta(
        self,
        *,
        session_id: str,
        profile: Optional[ChatSessionProfile],
        scene_picked: bool,
        module_scope: str,
        catalog_anchor: Sequence[str],
        add_asked_fields: Sequence[str] = (),
        add_suppressed_fields: Sequence[str] = (),
    ) -> None:
        base = dict(profile.interests) if profile and isinstance(profile.interests, dict) else {}
        session_meta = dict(base.get(_SESSION_META_KEY) or {})
        if scene_picked:
            session_meta["scene_picked"] = True
        if module_scope:
            session_meta["module_scope"] = module_scope
        if catalog_anchor:
            session_meta["catalog_anchor"] = list(catalog_anchor)
        if add_asked_fields:
            asked = {str(x).strip() for x in session_meta.get("asked_fields", []) if str(x).strip()}
            asked.update(str(x).strip() for x in add_asked_fields if str(x).strip())
            session_meta["asked_fields"] = sorted(asked)
        if add_suppressed_fields:
            suppressed = {str(x).strip() for x in session_meta.get("suppressed_fields", []) if str(x).strip()}
            suppressed.update(str(x).strip() for x in add_suppressed_fields if str(x).strip())
            session_meta["suppressed_fields"] = sorted(suppressed)
        session_meta["has_collected_fields"] = _lead_collected_fields(base.get(_LEAD_META_KEY) or {}, profile)
        base[_SESSION_META_KEY] = session_meta
        await self._profile_repo.upsert_profile(session_id=session_id, interests=base)


def _build_tagged_context(doc: _DocContext, *, is_primary: bool = True) -> str:
    body = (doc.body or "").strip()
    snippet = (doc.snippet or "").strip()
    chunk = body[:6000] if body else snippet[:2000]
    tag = "【主文档】" if is_primary else "【关联文档·可补充细节】"
    return f"{tag}{doc.title}\n文档标题：{doc.title}\n正文摘录：\n{chunk}"


def _infer_module_scope(anchor: Optional[CatalogNode]) -> str:
    if not anchor:
        return ""
    path = " / ".join(anchor.path_titles)
    title = anchor.title or ""
    if "使用指南" in path or "使用指南" in title:
        return "使用指南"
    if "案例分析" in path or "案例分析" in title:
        return "案例分析"
    if "平台介绍" in path or "平台介绍" in title:
        return "平台介绍"
    return ""


def _read_session_meta(profile: Optional[ChatSessionProfile]) -> Dict[str, Any]:
    if not profile or not isinstance(profile.interests, dict):
        return {}
    raw = profile.interests.get(_SESSION_META_KEY)
    return raw if isinstance(raw, dict) else {}


def _lead_collected_fields(lead_meta: Dict[str, Any], profile: Optional[ChatSessionProfile]) -> Dict[str, bool]:
    return {
        "name": bool(display_name_for_chat(profile)),
        "org_name": bool(lead_meta.get("org_name") or (profile.org_name if profile else "")),
        "contact": bool(lead_meta.get("contact_value")),
    }


def _session_field_sets(profile: Optional[ChatSessionProfile]) -> tuple[set[str], set[str]]:
    meta = _read_session_meta(profile)
    asked = {str(x).strip() for x in meta.get("asked_fields", []) if str(x).strip()}
    suppressed = {str(x).strip() for x in meta.get("suppressed_fields", []) if str(x).strip()}
    return asked, suppressed


def _build_v4_prompt(
    *,
    question: str,
    profile: Optional[ChatSessionProfile],
    catalog_path: str,
    dialog_level: int,
    related_titles: List[str],
    has_media: bool,
    history: Sequence[ChatMessageRow] = (),
    skill_instructions: str = "",
) -> str:
    name = display_name_for_chat(profile)
    vt = (profile.visitor_type if profile else "") or ""
    org = (profile.org_name if profile else "") or ""
    lead_meta = (
        profile.interests.get(_LEAD_META_KEY)
        if profile and isinstance(profile.interests, dict) and isinstance(profile.interests.get(_LEAD_META_KEY), dict)
        else {}
    )
    collected = _lead_collected_fields(lead_meta, profile)
    asked_fields, suppressed_fields = _session_field_sets(profile)
    missing_fields = [
        label
        for key, label in (
            ("name", "称呼"),
            ("org_name", "工作单位"),
            ("contact", "联系方式"),
        )
        if not collected.get(key) and key not in asked_fields and key not in suppressed_fields
    ]
    mem = []
    if name:
        mem.append(f"称呼：{name}")
    if vt:
        mem.append(f"身份（内部参考，勿复述）：{vt}")
    if org:
        mem.append(f"单位（内部参考，勿写出）：{org}")
    if lead_meta.get("contact_value"):
        mem.append("联系方式：已留存（禁止再次索要电话、微信或联系方式）")
    if lead_meta.get("interested_product"):
        mem.append(f"已关注产品：{lead_meta.get('interested_product')}")
    if asked_fields:
        asked_labels = "、".join(_FIELD_LABELS.get(key, key) for key in sorted(asked_fields))
        mem.append(f"已询问过字段（禁止再次追问）：{asked_labels}")
    if suppressed_fields:
        suppressed_labels = "、".join(_FIELD_LABELS.get(key, key) for key in sorted(suppressed_fields))
        mem.append(f"用户已指出不要重复追问这些字段（永久屏蔽）：{suppressed_labels}")
    wants_visual = any(k in (question or "") for k in ("有图", "看图", "图片", "截图", "示意图", "流程图"))
    media_hint = (
        "配图/视频已在界面展示；先用1句引导语承接图片或视频，再用2-3个要点说明素材中的内容和适用场景；"
        "如果上下文里出现“参考图1/参考图2/参考视频1”的识读摘要，请在回答里自然结合这些参考素材说明重点；"
        "禁止输出图片URL；禁止说「没有图片/无法展示图片」。"
        if has_media
        else ("用户想看图，若上下文含插图描述请用文字说明；若无素材再说明。" if wants_visual else "正文精炼（约90-150字，最多3段）。")
    )
    related = "、".join(related_titles[:3]) if related_titles else ""
    role_hint = _role_answer_hint(vt)
    hist_block = _format_history_for_prompt(history)
    progression_rule = _build_progression_rule(question=question, history=history, catalog_path=catalog_path)
    broad_overview_rule = _build_broad_overview_rule(question=question, history=history, catalog_path=catalog_path)
    persona_intro_rule = _build_persona_intro_rule(history=history)
    already_asked_identity = _history_has_identity_question(hist_block)
    identity_rule = (
        "" if not already_asked_identity
        else "注意：本轮对话已询问过学科/年级/身份，不要再重复追问这类信息。\n"
    )
    repeat_topic_rule = ""
    if hist_block and question.strip() in hist_block:
        repeat_topic_rule = "用户在重复同一问题，请在已讲内容基础上补充细节或案例，禁止原样复述前次回答。\n"
    skill_part = f"{skill_instructions}\n" if skill_instructions.strip() else ""
    next_field_rule = (
        f"本轮如需补充客户信息，最多只自然询问这一项：{missing_fields[0]}。\n"
        if missing_fields
        else "客户称呼、工作单位、联系方式已基本齐全；禁止再索要留资，只继续正常解答和引导下一步。\n"
    )
    persona_template = build_model_agnostic_sales_persona_template()
    return (
        f"{persona_template}\n"
        "你是有为人工智能教育平台的资深销售顾问，名字叫「小为顾问」，面向学校老师、机构负责人和家长沟通。\n"
        "人设：专业、真诚、有经验、有判断，不过度推销；会认真承接对方的话，像真人顾问一样自然交流。\n"
        "核心目标：让用户感觉是在和一位愿意帮助自己解决问题的顾问沟通，而不是知识库机器人、客服机器人、调查问卷或CRM信息收集机器人。\n"
        "表达要求：回答精简、口语化、像真人顾问；不要堆砌资料，不要机械复读标题，不要只罗列A方案/B方案/C方案。\n"
        "篇幅要求：单轮尽量控制在 3 段以内、2-4 个要点以内；能一句说清的，不要展开成一大段。\n"
        "不要写成产品宣讲稿，不要一上来总述“四大课程/四大模块”；优先围绕用户当前场景讲 2-4 个最相关点。\n"
        "先接住用户这句话，再展开；用户提供信息后，必须先反馈理解、给出观点或共鸣，再决定是否追问，不要立即进入下一轮提问。\n"
        "对话优先级：解决用户问题 > 顾问体验 > 销售线索收集；线索收集只能是咨询过程的自然结果。\n"
        "顾问分析机制：当已经获得较多需求信息时，必须按「需求总结 → 专业判断 → 推荐建议」推进，禁止继续追问。\n"
        "推荐表达：可以说「我更建议」「我会优先推荐」「从经验来看」「如果是我来规划」「根据您的情况」；不要说「A也可以、B也可以、看需求决定」。\n"
        "固定结构：\n"
        "- 第1段：先承接用户上一句，用1-2句自然回应；\n"
        "- 第2段：先给1句判断或建议，再用2-4条列表讲重点；\n"
        "- 第3段：用1句自然互动提问推进下一步；\n"
        "- 只有在确实缺信息时，才额外补1句信息采集问题，而且每轮最多主动收集一个字段。\n"
        "- 总起引导句和结尾互动句必须是正常段落，禁止放进列表；\n"
        "- 列表只用于拆分核心要点，禁止把整段对话都写成列表；\n"
        "- 核心要点优先写成 `- **关键词**：说明`，关键词必须加粗；\n"
        "- 避免大段正文，尽量让用户一眼扫清重点。\n"
        f"当前目录位置：{catalog_path}（对话层级 {dialog_level}）。\n"
        f"{skill_part}"
        "规则：\n"
        "- 只基于给定上下文回答；禁止编造目录外模块；\n"
        "- 禁止出现语雀/知识库/目录结构等字样；\n"
        "- 禁止说「不知道您是谁/看不到身份信息」——下方「称呼/身份」仅供你组织回答；\n"
        "- 不要输出「同模块里还可以看看：XX」这类固定模板句；\n"
        "- 每场对话中，学科/年级类信息最多询问1次；联系方式已留存则不再索要；\n"
        "- 整场对话留资/试用引导最多出现1次，且不在内容讲解中间插入；\n"
        "- 如果用户说“已经给过联系方式/电话”，必须承认已记录，并继续回答当前问题；禁止道歉后再次索要。\n"
        "- 禁止推荐当前目录节点之外的其他一级模块（如「智能招生」）；\n"
        "- 无需每轮都问好，不要重复前次寒暄。\n"
        "- 主动采集的客户字段只有三项：称呼、工作单位、联系方式；除此之外不主动收集其他个人信息；\n"
        "- 称呼要自然询问，例如“方便的话，我该怎么称呼您？”；禁止说“请问您贵姓/请填写姓名”；\n"
        "- 工作单位必须结合当前话题自然询问，例如“您这边是个人了解，还是代表单位在了解相关方案？”；禁止表单式询问；\n"
        "- 联系方式只能在完成需求分析、给出建议、建立信任并提供价值后询问；先说明可发送案例/资料/体验方式，再确认用户兴趣并自然询问联系方式；\n"
        "- 禁止直接问“手机号是多少/留个微信吧/请填写联系方式”；\n"
        "- 若只是初次泛泛咨询，优先理解学段、课堂场景、使用目标中的一个，而不是索要联系方式；\n"
        f"{identity_rule}"
        f"{repeat_topic_rule}"
        f"{progression_rule}"
        f"{broad_overview_rule}"
        f"{persona_intro_rule}"
        f"{next_field_rule}"
        f"{role_hint}\n"
        f"{media_hint}\n"
        + (f"可自然串联提及的同级话题（仅在用户主动询问时才提）：{related}。\n" if related else "")
        + ("\n".join(mem) + "\n" if mem else "")
        + (hist_block + "\n" if hist_block else "")
        + f"用户问题：{question}"
    )


def _format_history_for_prompt(history: Sequence[ChatMessageRow], *, limit: int = 8) -> str:
    lines: List[str] = []
    for row in list(history)[-limit:]:
        t = (row.content or "").strip()
        if not t:
            continue
        label = "访客" if row.role == "user" else "顾问"
        if len(t) > 220:
            t = t[:220] + "…"
        lines.append(f"{label}：{t}")
    if not lines:
        return ""
    return "近期对话（承接上下文，勿重复寒暄）：\n" + "\n".join(lines)


def _build_natural_guide(
    *,
    profile: Optional[ChatSessionProfile],
    state: CatalogDialogState,
    candidates: Sequence[CatalogNode],
    history: Sequence[ChatMessageRow],
) -> str:
    _ = history
    name = display_name_for_chat(profile)
    path = " / ".join(state.path_titles)
    titles = [c.title for c in candidates[:4] if c.title]
    if not titles:
        return _build_soft_clarify(profile=profile, state=state)

    if state.dialog_level <= 0 and not path:
        lead = f"{name}，" if name else ""
        intro = _platform_intro_for_role(profile)
        body = titles[0] if len(titles) == 1 else "、".join(titles)
        return (
            f"{lead}{intro}\n\n"
            f"结合您的情况，可以从这几块先了解：{body}。您想先聊哪一块？"
        )

    parent = path or "这一块"
    opts = "、".join(titles)
    lead = f"{name}，" if name and not _greeted_recently(history) else ""
    return f"{lead}在「{parent}」里，常见会先了解 {opts}。您更关心哪一块？"


def _build_soft_clarify(*, profile: Optional[ChatSessionProfile], state: CatalogDialogState) -> str:
    name = display_name_for_chat(profile)
    path = " / ".join(state.path_titles)
    lead = f"{name}，" if name else ""
    if path:
        return f"{lead}您还在看「{path}」相关内容，可以直接说您想深入了解的小点，我来给您讲。"
    return f"{lead}您可以直接说想了解的模块名称，例如平台介绍或使用指南里的某一功能。"


def _pick_primary_doc(docs: Sequence[_DocContext], *, anchor_title: str) -> Optional[_DocContext]:
    title = (anchor_title or "").strip()
    if not docs:
        return None
    for d in docs:
        if (d.title or "").strip() == title:
            return d
    return docs[0]


def _should_show_media(
    *,
    question: str,
    node: CatalogNode,
    primary: Optional[_DocContext],
    docs: Sequence[_DocContext] = (),
) -> bool:
    scope = list(docs) if docs else ([primary] if primary else [])
    if not scope or not any(_doc_body_has_media(d) for d in scope):
        return False
    q = (question or "").strip()
    if any(k in q for k in ("有图", "看图", "图片", "截图", "示意图", "流程图")):
        return True
    if any(k in q for k in ("看看", "介绍", "课程", "案例", "内容", "方案", "是什么", "有哪些", "有没有", "相关")):
        return True
    if any(k in q for k in ("测试账号", "试用账号", "申请测试", "开通试用")) and not any(
        k in q for k in ("介绍", "了解", "看看", "功能", "课程", "平台")
    ):
        return False
    return _title_match_score(q, node.title) >= 55 or _title_match_score(node.title, q) >= 55


def _doc_body_has_media(doc: _DocContext) -> bool:
    if doc_body_has_images(doc):
        return True
    bundle = collect_media_from_doc_contexts(
        [doc],
        question=doc.title or "",
        max_images=0,
        max_videos=1,
        primary_doc_title="",
    )
    return bool(bundle.videos)


def _should_skip_mcp_search(*, prefer_cached: bool, direct_doc_hits: int, hit_count: int) -> bool:
    if prefer_cached and direct_doc_hits > 0:
        return True
    if direct_doc_hits >= 2:
        return True
    if hit_count >= max(2, min(settings.chat_v15_max_docs, 4)):
        return True
    return False


def _media_scope_docs(
    *,
    question: str,
    primary: Optional[_DocContext],
    docs: Sequence[_DocContext],
) -> List[_DocContext]:
    q = (question or "").strip()
    if any(k in q for k in ("腾讯", "乐高", "索尼", "苹果", "案例", "视频", "图片", "课程", "内容", "通识", "项目", "软件", "硬件", "Swift", "Python", "算法")):
        return list(docs)
    if len(docs) > 1 and (not primary or not _doc_body_has_media(primary)):
        return list(docs)
    return [primary] if primary else list(docs[:1])


def _history_has_identity_question(hist_block: str) -> bool:
    """检查历史中是否已经询问过学科/年级等身份信息。"""
    triggers = ("学科", "年级", "带几年级", "教什么", "您是哪个", "负责哪个", "教哪个")
    return any(k in (hist_block or "") for k in triggers)


def _build_progression_rule(*, question: str, history: Sequence[ChatMessageRow], catalog_path: str) -> str:
    assistant_turns = [row for row in history if row.role == "assistant" and (row.content or "").strip()]
    if not assistant_turns:
        return ""

    covered = _covered_dimensions_from_history(history)
    covered_text = "、".join(covered[:4]) if covered else "通用卖点/基础介绍"
    focus = _current_need_focus(question=question, catalog_path=catalog_path)
    focus_text = focus or "用户本轮刚刚补充的关注点"
    follow_up = _looks_like_progressive_follow_up(question=question, history=history)

    if follow_up:
        return (
            "当前已进入多轮追问/细化讲解阶段：\n"
            f"- 近期已讲过：{covered_text}；禁止再按这些维度整段复述。\n"
            f"- 本轮只围绕「{focus_text}」往下讲一层，补充新的具体信息。\n"
            "- 若用户切换到新课程/新模块，可用1句承接切换，再直接讲新模块，不要回头重复旧模块卖点。\n"
            "- 若必须复用前文内容，只能压缩改写成1句承接，然后立刻补充新的细节、场景或建议。\n"
            "- 用户若刚回答了你的追问（如年级、课堂形式、教学目标、关注课程），必须把这当成新的限定条件继续讲，不要退回整套总介绍。\n"
        )

    return (
        "若本轮属于同一主题的继续沟通：\n"
        f"- 已讲内容主要包括：{covered_text}；不要原样复述。\n"
        f"- 继续围绕「{focus_text}」补充下一层细节，保持递进。\n"
    )


def _build_broad_overview_rule(*, question: str, history: Sequence[ChatMessageRow], catalog_path: str) -> str:
    q = (question or "").strip()
    if not q:
        return ""
    if any(row.role == "assistant" and (row.content or "").strip() for row in history):
        return ""
    specific_modules = ("乐高", "苹果", "索尼", "腾讯", "智能招生", "跨学科", "使用指南", "学校AI场景定制")
    if any(k in q for k in specific_modules):
        return ""
    broad_marks = ("通识教育", "通识课程", "课程内容", "整体方案", "平台内容", "内容", "介绍", "了解")
    path = (catalog_path or "").strip()
    if not (any(k in q for k in broad_marks) or "人工智能通识" in path):
        return ""
    return (
        "如果用户此轮仍在做整体了解、还没有点名具体产品方向：\n"
        "- 禁止逐条罗列全部四套方案，也不要把四套产品都展开成卖点清单；\n"
        "- 先用1句概括“这类方案主要差在什么”；\n"
        "- 再只按 2 个差异维度做列表，例如“动手搭建 vs 软件编程”或“老师上手 vs 项目延展”；\n"
        "- 若要举例，最多举 1 到 2 个代表方向，不要把所有产品都讲一遍；\n"
        "- 最后只问 1 个收敛问题，帮助用户判断下一步该看哪条线。\n"
    )


def _build_persona_intro_rule(*, history: Sequence[ChatMessageRow]) -> str:
    assistant_texts = [(row.content or "").strip() for row in history if row.role == "assistant" and (row.content or "").strip()]
    if not assistant_texts:
        return ""
    if any("我是小为顾问" in text or "小为顾问" in text for text in assistant_texts):
        return ""
    if len(assistant_texts) <= 1:
        return (
            "这是本会话的首个正式讲解回合：\n"
            "- 请在第1段自然带出一次你的身份，例如“我是小为顾问，我先帮您梳理一下”；\n"
            "- 只介绍这一次，后续轮次不要重复自报身份。\n"
        )
    return ""


def _covered_dimensions_from_history(history: Sequence[ChatMessageRow]) -> List[str]:
    text = "\n".join((row.content or "").strip() for row in history if row.role == "assistant")
    if not text:
        return []
    mapping = [
        ("适用学段", ("年级", "小学", "初中", "高中", "学段")),
        ("课堂场景", ("信息课", "社团", "机房", "平板", "课堂", "课后服务")),
        ("编程工具/载体", ("Swift", "Python", "图形化", "平板", "编程工具", "实验平台")),
        ("课堂落地/老师上手", ("备课", "课件", "搭环境", "教学平台", "师资培训", "上手")),
        ("算法逻辑/能力培养", ("算法", "逻辑", "判断", "循环", "思维", "问题解决")),
        ("项目案例/赛事延展", ("案例", "项目", "赛事", "展示", "成果")),
    ]
    picked: List[str] = []
    for label, keywords in mapping:
        if any(k in text for k in keywords):
            picked.append(label)
    return picked


def _current_need_focus(*, question: str, catalog_path: str) -> str:
    q = (question or "").strip()
    checks = [
        ("适用学段", ("年级", "小学", "初中", "高中", "五年级", "八年级")),
        ("课堂场景", ("信息课", "社团", "课后服务", "机房", "平板", "课堂")),
        ("软件编程", ("软件编程", "编程为主", "Swift", "Python", "代码")),
        ("算法逻辑", ("算法", "逻辑", "推导", "判断", "循环")),
        ("案例/演示", ("案例", "示例", "演示", "视频", "图片", "截图")),
        ("老师上手与落地", ("备课", "落地", "开课", "课时", "环境", "怎么上")),
    ]
    for label, keywords in checks:
        if any(k in q for k in keywords):
            return label
    path = (catalog_path or "").strip()
    if path:
        return path.split(" / ")[-1]
    return ""


def _looks_like_progressive_follow_up(*, question: str, history: Sequence[ChatMessageRow]) -> bool:
    q = (question or "").strip()
    if not q:
        return False
    assistant_turns = [row for row in history if row.role == "assistant" and (row.content or "").strip()]
    if not assistant_turns:
        return False
    markers = (
        "我看看",
        "我想看",
        "我想了解",
        "我更看重",
        "我更关注",
        "我目前",
        "我是",
        "我们是",
        "软件编程",
        "算法逻辑",
        "信息课",
        "社团",
        "年级",
    )
    if any(k in q for k in markers):
        return True
    last_assistant = (assistant_turns[-1].content or "").strip()
    if last_assistant and any(k in last_assistant for k in ("还是", "更偏向", "更看重", "主要是", "想先看", "方便说说")) and len(q) <= 24:
        return True
    if len(q) <= 18:
        return True
    return False


def _user_wants_more(question: str) -> bool:
    """用户主动表达想继续了解其它内容时才返回 True，避免每轮自动追加同模块推荐。"""
    q = (question or "").strip()
    triggers = ("还有", "其他", "别的", "更多", "还能", "还想", "再推荐", "推荐", "类似", "还有没有", "看看其他")
    return any(k in q for k in triggers)


def _build_repeat_feedback_apology(field: str) -> str:
    label = _FIELD_LABELS.get(field, "这项信息")
    return f"抱歉，刚才在{label}这项信息上重复确认了。我这边已经记住，后面不会再反复问您。"


def _detect_repeat_feedback_field(*, question: str, history: Sequence[ChatMessageRow]) -> str:
    q = (question or "").strip()
    if not q:
        return ""
    if not re.search(r"(重复|又问|还问|老问|一直问|反复问|都说了|不是说过|已经说过|刚说过)", q):
        return ""

    direct_rules = (
        ("org_name", ("单位", "学校", "机构")),
        ("contact", ("联系方式", "电话", "手机号", "手机", "微信")),
        ("name", ("姓名", "名字", "称呼")),
        ("interested_product", ("产品", "课程方向", "关注方向", "意向", "场景")),
    )
    for field, keywords in direct_rules:
        if any(k in q for k in keywords):
            return field

    for row in reversed(history):
        if row.role != "assistant":
            continue
        text = (row.content or "").strip()
        if not text:
            continue
        if any(k in text for k in ("怎么称呼您", "该怎么称呼您", "称呼您吗")):
            return "name"
        if any(k in text for k in ("单位或学校", "补充一下您的单位", "单位或机构", "您的单位")):
            return "org_name"
        if any(k in text for k in ("手机号或微信", "联系方式", "电话或微信", "留一个手机号")):
            return "contact"
        if any(k in text for k in ("产品或课程方向", "最想深入了解哪一个产品", "感兴趣产品")):
            return "interested_product"
    return ""


def _greeted_recently(history: Sequence[ChatMessageRow]) -> bool:
    for row in reversed(history[-3:]):
        if row.role == "assistant" and (row.content or "").strip():
            return True
    return False


def _platform_intro_for_role(profile: Optional[ChatSessionProfile]) -> str:
    vt = (profile.visitor_type if profile else "") or ""
    base = (
        "有为人工智能教育平台面向学校与机构，覆盖 AI 通识课、跨学科项目式学习（IDEAS-PBL）、"
        "教师端教学工具与案例社区，帮助把「学 AI」和「用 AI」落到日常教学里。"
    )
    if vt == "parent":
        return (
            base
            + " 对家长来说，更常关注的是孩子学什么、如何衔接升学与竞赛，以及怎样选适合年龄段的课程。"
        )
    if vt == "teacher":
        return base + " 老师一般会关心备课上课、作业评价和课堂落地是否省心。"
    return base


def _role_answer_hint(visitor_type: str) -> str:
    vt = (visitor_type or "").strip()
    if vt == "parent":
        return "回答视角：家长关注孩子升学与课程选择，语气务实。"
    if vt == "teacher":
        return "回答视角：教师关注课堂落地与教学流程。"
    if vt == "institution_decision_maker":
        return "回答视角：校方关注方案、部署与成效。"
    return ""


def _strip_media_urls_from_text(text: str, media: ChatMediaBundle) -> str:
    out = text or ""
    for img in media.images:
        u = (img.url or "").strip()
        if u:
            out = out.replace(u, "")
    out = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", out)
    out = re.sub(r"https?://\S+\.(?:png|jpg|jpeg|gif|webp|svg)(?:\?\S*)?", "", out, flags=re.I)
    return re.sub(r"\n{3,}", "\n\n", out).strip()
