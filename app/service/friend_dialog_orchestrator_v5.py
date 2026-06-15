from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, AsyncIterator, List, Optional, Sequence

from app.conversation.contact_extractor import extract_contact
from app.conversation.friend_persona_v5 import build_friend_v5_system_prompt, scene_description
from app.conversation.skill_catalog import SKILL_CATALOG, SkillRoute
from app.conversation.profile_extractor import ProfileExtractor
from app.core.logger import get_logger
from app.db.repositories import ChatMessageRow

logger = get_logger(__name__)
from app.rag.friend_v5_generator import FriendV5Generator
from app.rag.skill_router import route_skill
from app.schemas.chat import ChatMediaBundle, MediaItem
from app.schemas.chat_v5 import ChatV5DonePayload, FriendV5SourceItem
from app.service.friend_v5_tags import (
    SCENE_TO_TOC_TITLE,
    FriendV5TagStreamFilter,
    classify_friend_v5_tag,
    explore_product_tag_for_title,
    explore_product_title_from_tag,
    fallback_tags_for_scene,
    scene_for_toc_title,
    price_tag_for_scene,
    product_title_from_tag,
    toc_title_for_scene,
    trial_tag_for_scene,
    try_product_title_from_tag,
)
from app.service.friend_v5_yuque_deep_reader import (
    FriendV5YuqueDeepReadResult,
    should_deep_read_yuque_doc,
)


_MANUAL_YUQUE_HINT_RE = re.compile(
    r"(课程|产品|方案|指南|文档|手册|介绍|案例|乐高|AI|人工智能|项目化|招生|实验室|校本)"
)
_FOLLOWUP_CONFIRM_RE = re.compile(r"^(需要|继续|好|好的|可以|想了解|详细说说|展开|讲讲|要|嗯|行)(?:[，。,.!！?？\s].*)?$")
_FOLLOWUP_TOPIC_RE = re.compile(r"需要我(?:和你|帮你)?详细介绍(.+?)的内容吗")
_CASE_SECTION_TITLE = "案例与社区"
_CASE_LIBRARY_TITLE = "优秀案例库"
_PLATFORM_SECTION_TITLE = "平台介绍"
_GUIDE_SOURCES_HINT = "使用指南中的详细操作链接已整理在下方参考资料，您可点击查阅。"
_CASE_KB_FALLBACK_ANSWER = (
    "目前在上海、江苏、成都多所K12学校均有落地实施的具体案例，"
    "需了解更为具体的案例介绍，方便的话可以留下您的联系方式。"
)


@dataclass(frozen=True)
class _CatalogTagResult:
    tags: List[str]
    focus_node: Optional[dict[str, Any]]


@dataclass(frozen=True)
class _TagRhythmResult:
    tags: List[str]
    conversion_state: dict[str, Any]


@dataclass(frozen=True)
class _TagRouteResult:
    kind: str
    target_title: str = ""
    focus_node: Optional[dict[str, Any]] = None


class FriendDialogOrchestratorV5:
    def __init__(
        self,
        *,
        generator: FriendV5Generator,
        profile_repo: Any,
        yuque_search: Optional[Any] = None,
        scene_query_rewriter: Optional[Any] = None,
        profile_extractor: Optional[ProfileExtractor] = None,
        yuque_deep_reader: Optional[Any] = None,
        toc_nodes: Optional[Sequence[dict[str, Any]]] = None,
        yuque_url_limit: int = 3,
        require_web_sources: bool = True,
        admin_video_repository: Optional[Any] = None,
    ) -> None:
        self._generator = generator
        self._profile_repo = profile_repo
        self._yuque_search = yuque_search
        self._scene_query_rewriter = scene_query_rewriter
        self._yuque_deep_reader = yuque_deep_reader
        self._profile_extractor = profile_extractor or ProfileExtractor()
        self._toc_nodes = _normalize_toc_nodes(toc_nodes or [])
        self._yuque_url_limit = max(0, int(yuque_url_limit or 0))
        self._require_web_sources = bool(require_web_sources)
        self._admin_video_repository = admin_video_repository

    async def answer_stream(
        self,
        *,
        question: str,
        session_id: str,
        scene: str,
        trigger_type: str,
        history: Sequence[ChatMessageRow],
    ) -> AsyncIterator[dict[str, Any]]:
        yield _stage("profile", "小为正在记住这次对话里的关键信息...")
        profile = await self._profile_repo.get_profile(session_id=session_id)
        update = await self._profile_extractor.extract_update(
            question=question,
            history=history,
            current_profile=profile,
        )
        if update.display_name or update.org_name or update.interests or update.visitor_type:
            await self._profile_repo.upsert_profile(
                session_id=session_id,
                display_name=update.display_name,
                visitor_type=update.visitor_type,
                org_name=update.org_name,
                interests=update.interests,
            )
            profile = await self._profile_repo.get_profile(session_id=session_id)

        profile = await _persist_contact_from_question(
            profile_repo=self._profile_repo,
            session_id=session_id,
            question=question,
            profile=profile,
            scene=scene,
        )

        deep_read = FriendV5YuqueDeepReadResult()
        scene_query_rewrite: dict[str, Any] = {}
        scene_query = (question or "").strip()
        scene_case_continuation = False
        scene_guide_continuation = False
        if trigger_type == "scene":
            scene_query = toc_title_for_scene(scene)
            catalog_focus = _resolve_platform_intro_focus(
                product_title=scene_query,
                toc_nodes=self._toc_nodes,
            )
            if catalog_focus is None:
                catalog_focus = _best_toc_match(
                    question=scene_query,
                    scene="",
                    parsed_tags=[],
                    toc_nodes=self._toc_nodes,
                )
        else:
            catalog_focus = _best_toc_match(question=question, scene="", parsed_tags=[], toc_nodes=self._toc_nodes)
        tag_route = _resolve_tag_route(
            question=question,
            scene=scene,
            trigger_type=trigger_type,
            history=history,
            toc_nodes=self._toc_nodes,
        )
        followup_topic = _followup_topic_for_question(question=question, trigger_type=trigger_type, history=history)
        mcp_route: dict[str, Any] = {"mode": "none"}
        case_product_switch = False
        guide_product_switch = False
        cross_scene_product = (
            trigger_type in {"manual", "tag"}
            and _is_bare_product_question(question, scene=scene)
            and _is_other_scene_root_product(question, scene=scene)
        )
        if cross_scene_product:
            product_title = _product_title_from_question(question, scene=scene)
            answer = _cross_scene_redirect_answer(scene=scene, product=product_title)
            tags_result = _apply_recommendation_tag_rhythm(
                content_tags=[],
                question=question,
                scene=scene,
                trigger_type=trigger_type,
                history=history,
                route_kind="none",
                focus_node=None,
                toc_nodes=self._toc_nodes,
            )
            yield {"event": "token", "data": {"token": answer}}
            payload = ChatV5DonePayload(
                answer=answer,
                tags=tags_result.tags,
                sources=[],
                search_keywords=_derive_search_keywords(question),
                media=deep_read.media,
                profile_fields=_profile_fields(profile),
                fallback_used=False,
                debug={
                    "pipeline": "friend_v5",
                    "scene": scene,
                    "trigger_type": trigger_type,
                    "cross_scene_redirect": True,
                    "redirect_product": product_title,
                    "tag_route": _tag_route_debug(tag_route),
                    "conversion_state": tags_result.conversion_state,
                },
            )
            yield {"event": "done", "data": payload.model_dump()}
            return

        if (
            trigger_type != "scene"
            and _history_has_seen_case(history)
            and _is_bare_product_question(question, scene=scene)
            and _product_matches_scene(question, scene=scene)
        ):
            case_focus, case_query, is_case = _resolve_case_library_focus(
                product_title=_product_title_from_question(question, scene=scene),
                scene=scene,
                history=history,
                toc_nodes=self._toc_nodes,
            )
            if is_case and case_focus is not None:
                catalog_focus = case_focus
                scene_query = case_query
                scene_case_continuation = True
                case_product_switch = True
        if (
            trigger_type != "scene"
            and not case_product_switch
            and _history_active_content_branch(history) == "guide"
            and _is_bare_product_question(question, scene=scene)
            and _product_matches_scene(question, scene=scene)
        ):
            guide_focus, guide_query, is_guide = _resolve_guide_focus(
                product_title=_product_title_from_question(question, scene=scene),
                toc_nodes=self._toc_nodes,
            )
            if is_guide and guide_focus is not None:
                catalog_focus = guide_focus
                scene_query = guide_query
                scene_guide_continuation = True
                guide_product_switch = True
        if tag_route.focus_node is not None and not case_product_switch and not guide_product_switch:
            catalog_focus = tag_route.focus_node
            scene_query = tag_route.target_title or question
        elif followup_topic:
            scene_query = followup_topic
            catalog_focus = _best_toc_match(
                question=followup_topic,
                scene=scene,
                parsed_tags=[],
                toc_nodes=self._toc_nodes,
            )
        case_branch_used = False
        skill_route: Optional[SkillRoute] = None

        if tag_route.kind == "price":
            answer = _price_handoff_answer(scene=scene, profile=profile)
            tags_result = _apply_recommendation_tag_rhythm(
                content_tags=[],
                question=question,
                scene=scene,
                trigger_type=trigger_type,
                history=history,
                route_kind=tag_route.kind,
                focus_node=catalog_focus,
                toc_nodes=self._toc_nodes,
            )
            yield {"event": "token", "data": {"token": answer}}
            payload = ChatV5DonePayload(
                answer=answer,
                tags=tags_result.tags,
                sources=[],
                search_keywords=_derive_search_keywords(question),
                media=deep_read.media,
                profile_fields=_profile_fields(profile),
                fallback_used=False,
                debug={
                    "pipeline": "friend_v5",
                    "scene": scene,
                    "trigger_type": trigger_type,
                    "tag_route": _tag_route_debug(tag_route),
                    "next_followup_topic": None,
                    "skill_route": None,
                    "mcp_route": {"mode": "price_direct"},
                    "doc_deep_read_used": False,
                    "doc_deep_read": deep_read.debug,
                    "case_branch_used": False,
                    "conversion_state": tags_result.conversion_state,
                    "search_keyword_count": 1,
                    "web_source_count": 0,
                    "yuque_source_count": 0,
                    "catalog_tag_source": "fixed_v5_navigation",
                    "catalog_tag_node_count": len(self._toc_nodes),
                    "catalog_focus_node": _catalog_focus_debug(catalog_focus),
                },
            )
            yield {"event": "done", "data": payload.model_dump()}
            return

        if trigger_type == "scene" and self._scene_query_rewriter and not catalog_focus:
            yield _stage("scene_rewrite", "小为正在把你的场景需求改写成更贴近语雀目录的检索词...")
            try:
                rewritten_query = await self._scene_query_rewriter.rewrite(
                    question=question,
                    scene=scene,
                    toc_nodes=self._toc_nodes,
                )
            except Exception:
                rewritten_query = ""
            if rewritten_query:
                scene_query = rewritten_query
                if not catalog_focus:
                    catalog_focus = _best_toc_match(
                        question=scene_query,
                        scene=scene,
                        parsed_tags=[],
                        toc_nodes=self._toc_nodes,
                    )
            scene_query_rewrite = {
                "used": bool(rewritten_query),
                "rewritten_query": scene_query,
            }
        elif trigger_type == "scene":
            scene_query_rewrite = {
                "used": False,
                "rewritten_query": scene_query,
                "skipped": "fixed_scene_toc_mapping",
            }

        if (
            scene_case_continuation
            and not deep_read.used
            and self._yuque_deep_reader
            and catalog_focus
        ):
            case_branch_used = True
            mcp_route = _mcp_route_debug("scene_case_library", scene_query, catalog_focus)
            yield _stage("yuque_case_read", "小为正在匹配案例库里的真实案例...")
            try:
                deep_read = await _read_deep_by_focus_or_query(
                    reader=self._yuque_deep_reader,
                    focus=catalog_focus,
                    question=_case_query_context(question=scene_query or question, scene=scene, history=history),
                )
            except Exception:
                deep_read = FriendV5YuqueDeepReadResult(debug={"mode": "case_deep_read_error"})

        if trigger_type == "scene" and self._yuque_deep_reader and scene_query and not deep_read.used:
            mcp_route = _mcp_route_debug(
                _scene_mcp_mode(
                    scene_case_continuation=scene_case_continuation,
                    scene_guide_continuation=scene_guide_continuation,
                ),
                scene_query,
                catalog_focus,
            )
            yield _stage("yuque_deep_read", "小为正在读取语雀文档正文和图文视频...")
            try:
                deep_read = await _read_deep_by_focus_or_query(
                    reader=self._yuque_deep_reader,
                    focus=catalog_focus,
                    question=scene_query,
                )
            except Exception:
                deep_read = FriendV5YuqueDeepReadResult(debug={"mode": "deep_read_error"})

        if (
            not deep_read.used
            and tag_route.kind == "case"
            and self._yuque_deep_reader
            and tag_route.focus_node
        ):
            case_branch_used = True
            case_focus = tag_route.focus_node
            catalog_focus = case_focus
            mcp_route = _mcp_route_debug("tag_case_library", tag_route.target_title or question, case_focus)
            yield _stage("yuque_case_read", "小为正在匹配案例库里的真实案例...")
            try:
                deep_read = await _read_deep_by_focus_or_query(
                    reader=self._yuque_deep_reader,
                    focus=case_focus,
                    question=_case_query_context(
                        question=tag_route.target_title or question,
                        scene=tag_route.target_title or scene,
                        history=history,
                    ),
                    allow_search_fallback=False,
                )
            except Exception:
                deep_read = FriendV5YuqueDeepReadResult(debug={"mode": "case_deep_read_error"})

        if (
            not deep_read.used
            and trigger_type == "tag"
            and self._yuque_deep_reader
            and catalog_focus
            and tag_route.kind != "case"
        ):
            mcp_route = _mcp_route_debug(f"tag_{tag_route.kind or 'toc'}", scene_query or question, catalog_focus)
            yield _stage("yuque_deep_read", "小为正在读取语雀文档正文和图文视频...")
            try:
                deep_read = await _read_deep_by_focus_or_query(
                    reader=self._yuque_deep_reader,
                    focus=catalog_focus,
                    question=scene_query or question,
                )
            except Exception:
                deep_read = FriendV5YuqueDeepReadResult(debug={"mode": "deep_read_error"})

        if (
            not deep_read.used
            and trigger_type == "manual"
            and self._yuque_deep_reader
            and catalog_focus
            and (followup_topic or self._should_lookup_yuque(question=question, trigger_type=trigger_type))
        ):
            mcp_route = _mcp_route_debug("followup_topic" if followup_topic else "manual_toc", scene_query or question, catalog_focus)
            yield _stage("yuque_deep_read", "小为正在读取语雀文档正文和图文视频...")
            try:
                deep_read = await _read_deep_by_focus_or_query(
                    reader=self._yuque_deep_reader,
                    focus=catalog_focus,
                    question=scene_query or question,
                )
            except Exception:
                deep_read = FriendV5YuqueDeepReadResult(debug={"mode": "deep_read_error"})

        if not deep_read.used and followup_topic and self._yuque_deep_reader:
            mcp_route = {"mode": "followup_search", "query": followup_topic, "focus_node": None}
            yield _stage("yuque_deep_read", "小为正在读取语雀文档正文和图文视频...")
            try:
                deep_read = await _read_deep_by_focus_or_query(
                    reader=self._yuque_deep_reader,
                    focus=None,
                    question=followup_topic,
                )
            except Exception:
                deep_read = FriendV5YuqueDeepReadResult(debug={"mode": "followup_deep_read_error"})

        if not deep_read.used and self._should_deep_read_yuque(question=question, trigger_type=trigger_type):
            mcp_route = _mcp_route_debug("auto_deep_read", question, catalog_focus)
            yield _stage("yuque_deep_read", "小为正在读取语雀文档正文和图文视频...")
            try:
                deep_read = await _read_deep_by_focus_or_query(
                    reader=self._yuque_deep_reader,
                    focus=catalog_focus,
                    question=question,
                )
            except Exception:
                deep_read = FriendV5YuqueDeepReadResult(debug={"mode": "deep_read_error"})

        yuque_sources: List[FriendV5SourceItem] = list(deep_read.sources)
        if not deep_read.used and self._should_lookup_yuque(question=question, trigger_type=trigger_type):
            yield _stage("yuque_links", "小为正在找相关语雀补充阅读入口...")
            yuque_sources = await self._lookup_yuque_sources(question)

        skill_route = route_skill(question)
        if skill_route is None and deep_read.used:
            skill_route = SKILL_CATALOG.get("smart-summary")

        case_intent = case_branch_used or scene_case_continuation or tag_route.kind == "case"
        effective_scene = scene
        if tag_route.target_title:
            mapped_scene = scene_for_toc_title(tag_route.target_title)
            if mapped_scene:
                effective_scene = mapped_scene

        if case_intent and not deep_read.used:
            answer = _case_kb_fallback_answer()
            tags_result = _apply_recommendation_tag_rhythm(
                content_tags=[],
                question=question,
                scene=effective_scene,
                trigger_type=trigger_type,
                history=history,
                route_kind="case",
                focus_node=catalog_focus,
                toc_nodes=self._toc_nodes,
            )
            yield {"event": "token", "data": {"token": answer}}
            payload = ChatV5DonePayload(
                answer=answer,
                tags=tags_result.tags,
                sources=[],
                search_keywords=_derive_search_keywords(question),
                media=deep_read.media,
                profile_fields=_profile_fields(profile),
                fallback_used=True,
                debug={
                    "pipeline": "friend_v5",
                    "scene": scene,
                    "trigger_type": trigger_type,
                    "scene_query_rewrite": scene_query_rewrite,
                    "tag_route": _tag_route_debug(tag_route),
                    "scene_case_continuation": scene_case_continuation,
                    "scene_guide_continuation": scene_guide_continuation,
                    "case_product_switch": case_product_switch,
                    "guide_product_switch": guide_product_switch,
                    "web_search_fallback_enabled": False,
                    "skill_route": None,
                    "mcp_route": mcp_route,
                    "doc_deep_read_used": False,
                    "doc_deep_read": deep_read.debug,
                    "case_toc_miss": tag_route.kind == "case" and not deep_read.used,
                    "case_branch_used": case_branch_used,
                    "case_kb_fallback": True,
                    "conversion_state": tags_result.conversion_state,
                    "search_keyword_count": len(_derive_search_keywords(question)),
                    "web_source_count": 0,
                    "yuque_source_count": 0,
                    "catalog_tag_source": "fixed_v5_navigation",
                    "catalog_tag_node_count": len(self._toc_nodes),
                    "catalog_focus_node": _catalog_focus_debug(catalog_focus),
                },
            )
            yield {"event": "done", "data": payload.model_dump()}
            return

        yield _stage("searching", "小为正在结合语雀资料整理回答...")
        system_prompt = build_friend_v5_system_prompt()
        case_answer_mode = case_intent and deep_read.used
        user_prompt = self._build_user_prompt(
            question=question,
            scene=effective_scene,
            trigger_type=trigger_type,
            history=history,
            profile=profile,
            yuque_sources=yuque_sources,
            deep_read=deep_read,
            skill_route=skill_route,
            case_answer_mode=case_answer_mode,
            case_toc_miss=tag_route.kind == "case" and not deep_read.used,
            product_focus=tag_route.target_title or "",
        )
        parser = FriendV5TagStreamFilter(scene=scene)
        answer_parts: List[str] = []
        web_sources: List[FriendV5SourceItem] = []
        search_keywords: List[str] = []

        # 语雀深读未命中时，开启模型联网搜索作为兜底
        enable_web_search = not deep_read.used
        try:
            stream_iter = self._generator.stream(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                enable_search=enable_web_search,
            )
        except TypeError:
            stream_iter = self._generator.stream(system_prompt=system_prompt, user_prompt=user_prompt)
        async for item in stream_iter:
            if item.event == "web_sources":
                web_sources.extend(_dedupe_sources(item.sources, existing=web_sources))
                continue
            if item.event == "search_keywords":
                search_keywords = _dedupe_keywords([*search_keywords, *list(item.search_keywords or [])])
                continue
            if item.event == "token":
                visible = parser.feed(item.token)
                if visible:
                    answer_parts.append(visible)
                    yield {"event": "token", "data": {"token": visible}}

        parsed = parser.finish()
        answer = parsed.answer or "".join(answer_parts).strip()
        answer = _strip_inline_urls(answer)

        source_items = _dedupe_sources(_source_urls_to_items(parsed.source_urls), existing=web_sources)
        if source_items:
            logger.info("V5 从 [SOURCES] 块解析到 %d 个来源链接", len(source_items))
        if web_sources:
            logger.info("V5 从联网搜索响应解析到 %d 个来源链接", len(web_sources))
        merged_sources = _dedupe_sources([*web_sources, *source_items, *yuque_sources], existing=[])
        merged_keywords = search_keywords or _derive_search_keywords(question)

        if (tag_route.kind == "guide" or scene_guide_continuation) and answer:
            answer = _append_guide_inline_link(answer, merged_sources, scene=scene)

        catalog_tags = _catalog_matched_tags(
            question=question,
            scene=scene,
            parsed_tags=parsed.tags,
            toc_nodes=self._toc_nodes,
            focus=catalog_focus,
        )
        next_followup_topic = _select_next_followup_topic(
            focus=catalog_tags.focus_node,
            toc_nodes=self._toc_nodes,
            deep_read=deep_read,
            answer=answer,
            history=history,
        )
        followup_sibling_topic = _select_followup_sibling_topic(
            focus=catalog_tags.focus_node,
            toc_nodes=self._toc_nodes,
            history=history,
            exclude=(next_followup_topic,),
        )
        answer = _append_followup_question(
            _strip_existing_followup_questions(answer),
            next_followup_topic,
            sibling_topic=followup_sibling_topic,
        )
        rhythm_tags = _apply_recommendation_tag_rhythm(
            content_tags=catalog_tags.tags,
            question=question,
            scene=scene,
            trigger_type=trigger_type,
            history=history,
            route_kind=tag_route.kind,
            focus_node=catalog_tags.focus_node,
            toc_nodes=self._toc_nodes,
        )
        media_suppressed = _should_suppress_initial_media(trigger_type=trigger_type, tag_route=tag_route)
        admin_scene_media = await _load_admin_scene_media(
            repository=self._admin_video_repository,
            scene=scene,
            trigger_type=trigger_type,
            history=history,
        )
        display_media = _merge_media(
            ChatMediaBundle() if media_suppressed else deep_read.media,
            admin_scene_media,
        )

        payload = ChatV5DonePayload(
            answer=answer,
            tags=rhythm_tags.tags,
            sources=merged_sources,
            search_keywords=merged_keywords,
            media=display_media,
            profile_fields=_profile_fields(profile),
            fallback_used=False,
            debug={
                "pipeline": "friend_v5",
                "scene": scene,
                "trigger_type": trigger_type,
                "scene_query_rewrite": scene_query_rewrite,
                "tag_route": _tag_route_debug(tag_route),
                "scene_case_continuation": scene_case_continuation,
                "scene_guide_continuation": scene_guide_continuation,
                "case_product_switch": case_product_switch,
                "guide_product_switch": guide_product_switch,
                "next_followup_topic": next_followup_topic,
                "followup_sibling_topic": followup_sibling_topic,
                "web_search_fallback_enabled": enable_web_search,
                "skill_route": _skill_route_debug(skill_route),
                "mcp_route": mcp_route,
                "doc_deep_read_used": bool(deep_read.used),
                "media_suppressed": media_suppressed,
                "admin_scene_video_count": len(admin_scene_media.videos),
                "doc_deep_read": deep_read.debug,
                "case_toc_miss": tag_route.kind == "case" and not deep_read.used,
                "case_branch_used": case_branch_used,
                "conversion_state": rhythm_tags.conversion_state,
                "search_keyword_count": len(merged_keywords),
                "web_source_count": len([item for item in merged_sources if item.source_type == "web"]),
                "yuque_source_count": len([item for item in merged_sources if item.source_type == "yuque"]),
                "catalog_tag_source": "fixed_v5_navigation",
                "catalog_tag_node_count": len(self._toc_nodes),
                "catalog_focus_node": _catalog_focus_debug(catalog_tags.focus_node),
            },
        )
        yield {"event": "done", "data": payload.model_dump()}

    def _build_user_prompt(
        self,
        *,
        question: str,
        scene: str,
        trigger_type: str,
        history: Sequence[ChatMessageRow],
        profile: Any,
        yuque_sources: Sequence[FriendV5SourceItem],
        deep_read: FriendV5YuqueDeepReadResult,
        skill_route: Optional[SkillRoute],
        case_answer_mode: bool = False,
        case_toc_miss: bool = False,
        product_focus: str = "",
    ) -> str:
        hist_lines: List[str] = []
        for row in list(history)[-8:]:
            role = "用户" if row.role == "user" else "小为"
            content = (row.content or "").strip()
            if content:
                hist_lines.append(f"{role}：{content[:240]}")
        yuque_lines = [
            f"- {item.title}: {item.url}"
            for item in yuque_sources
            if (item.title or "").strip() and (item.url or "").strip()
        ]
        if deep_read.used:
            yuque_instruction = "已提供语雀文档正文摘录，请优先基于该摘录回答；如果摘录没有提到，不要编造。"
        elif case_toc_miss:
            product_hint = (product_focus or scene).strip()
            yuque_instruction = (
                f"语雀「优秀案例库」中未找到「{product_hint}」的同名案例文档。"
                "请联网搜索该产品/方向的落地案例或应用场景后回答，并如实说明资料来自联网检索；"
                "不要编造语雀里已有的其他产品案例，也不要展示无关产品的图片。"
            )
        else:
            yuque_instruction = (
                "请联网搜索后回答。你可以把语雀链接作为补充阅读入口，但不要声称已经读过链接里的全文。"
            )
        skill_block = (
            f"【本轮 Skill】\n{skill_route.skill_id}\n{skill_route.generation_instruction}\n\n"
            if skill_route
            else ""
        )
        deep_block = _case_focused_prompt_block(deep_read.prompt_block) if case_answer_mode and deep_read.used else deep_read.prompt_block
        case_mode_block = (
            "【优秀案例库模式】优先讲学校/机构的落地实践（校名、学段、学生项目过程与成果），"
            "用案例故事帮助判断；不要写成产品功能清单或系统能力介绍，除非用户明确追问平台能力。\n\n"
            if case_answer_mode
            else ""
        )
        return (
            f"【当前场景】\n{scene}\n{scene_description(scene)}\n\n"
            f"【触发方式】\n{trigger_type}\n\n"
            f"{case_mode_block}"
            f"【已了解的信息】\n{_profile_block(profile)}\n\n"
            f"【最近对话】\n{chr(10).join(hist_lines) if hist_lines else '（暂无）'}\n\n"
            f"【语雀补充阅读链接】\n{chr(10).join(yuque_lines) if yuque_lines else '（本轮不提供语雀链接）'}\n\n"
            f"{deep_block + chr(10) + chr(10) if deep_read.used else ''}"
            f"{skill_block}"
            f"【用户这轮想了解】\n{question}\n\n"
            f"{yuque_instruction}"
        )

    def _should_deep_read_yuque(self, *, question: str, trigger_type: str) -> bool:
        return bool(self._yuque_deep_reader and should_deep_read_yuque_doc(question=question, trigger_type=trigger_type))

    def _should_lookup_yuque(self, *, question: str, trigger_type: str) -> bool:
        if not self._yuque_search or self._yuque_url_limit <= 0:
            return False
        if trigger_type == "scene":
            return False
        if trigger_type == "tag":
            return True
        return bool(_MANUAL_YUQUE_HINT_RE.search(question or ""))

    async def _lookup_yuque_sources(self, query: str) -> List[FriendV5SourceItem]:
        if not self._yuque_search:
            return []
        try:
            if hasattr(self._yuque_search, "search_docs"):
                try:
                    raw_items = await self._yuque_search.search_docs(query=query, limit=self._yuque_url_limit)
                except TypeError:
                    raw_items = await self._yuque_search.search_docs(query)
            elif hasattr(self._yuque_search, "search"):
                raw_items = await self._yuque_search.search(query)
            else:
                raw_items = []
        except Exception:
            raw_items = []
        out: List[FriendV5SourceItem] = []
        for idx, raw in enumerate(list(raw_items or [])[: self._yuque_url_limit], start=1):
            title = str(getattr(raw, "title", "") or _get(raw, "title") or "").strip()
            url = str(getattr(raw, "url", "") or _get(raw, "url") or "").strip()
            snippet = str(getattr(raw, "snippet", "") or getattr(raw, "summary", "") or _get(raw, "summary") or "").strip()
            doc_id = str(getattr(raw, "doc_id", "") or _get(raw, "doc_id") or "").strip()
            if not title and not url:
                continue
            out.append(
                FriendV5SourceItem(
                    source_type="yuque",
                    title=title or url,
                    url=url or None,
                    snippet=snippet or None,
                    doc_id=doc_id or None,
                    index=idx,
                )
            )
        return out


def _stage(stage: str, detail: str) -> dict[str, Any]:
    return {"event": "stage", "data": {"stage": stage, "detail": detail}}


def _profile_fields(profile: Any) -> dict[str, Any]:
    if profile is None:
        return {}
    out: dict[str, Any] = {}
    for key in ("display_name", "org_name", "interests"):
        value = getattr(profile, key, None)
        if value:
            out[key] = value
    return out


def _profile_block(profile: Any) -> str:
    fields = _profile_fields(profile)
    if not fields:
        return "（暂无）"
    lines: List[str] = []
    if fields.get("display_name"):
        lines.append(f"称呼：{fields['display_name']}")
    if fields.get("org_name"):
        lines.append(f"单位：{fields['org_name']}")
    interests = fields.get("interests")
    if isinstance(interests, dict):
        lead = interests.get("_lead")
        if isinstance(lead, dict):
            contact = str(lead.get("contact_value") or "").strip()
            if contact:
                lines.append(f"联系方式：{contact}（已收集，勿重复索要）")
            product = str(lead.get("interested_product") or "").strip()
            if product:
                lines.append(f"感兴趣产品：{product}")
    if interests and not lines:
        lines.append(f"兴趣参考：{interests}")
    return "\n".join(lines) if lines else "（暂无）"


def _source_urls_to_items(urls: list[str]) -> list[FriendV5SourceItem]:
    items: list[FriendV5SourceItem] = []
    seen: set[str] = set()
    for idx, url in enumerate(urls, start=1):
        u = url.strip()
        if not u or not u.startswith("http") or u in seen:
            continue
        seen.add(u)
        items.append(
            FriendV5SourceItem(
                source_type="web",
                title=f"参考资料 {idx}",
                url=u,
                index=idx,
            )
        )
    return items


def _dedupe_sources(
    sources: Sequence[FriendV5SourceItem],
    *,
    existing: Sequence[FriendV5SourceItem],
) -> List[FriendV5SourceItem]:
    seen = {_source_dedupe_key(item) for item in existing if _source_dedupe_key(item)}
    out: List[FriendV5SourceItem] = []
    next_index = 1
    for item in existing:
        if getattr(item, "index", None):
            try:
                next_index = max(next_index, int(item.index) + 1)
            except Exception:
                pass
    for raw_item in sources:
        item = FriendV5SourceItem.model_validate(raw_item)
        key = _source_dedupe_key(item)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item.model_copy(update={"index": next_index}))
        next_index += 1
    return out


def _source_dedupe_key(item: FriendV5SourceItem) -> str:
    url = (item.url or "").strip().lower()
    if url:
        return url
    title = (item.title or "").strip().lower()
    source_type = (item.source_type or "").strip().lower()
    if title:
        return f"{source_type}:{title}"
    return ""


def _dedupe_keywords(items: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for raw in items:
        for item in re.split(r"[\n,，、;；|]+", str(raw or "").strip()):
            keyword = item.strip().strip("\"'“”‘’")
            if not keyword or keyword in seen:
                continue
            seen.add(keyword)
            out.append(keyword[:120])
    return out


def _derive_search_keywords(question: str) -> List[str]:
    return _dedupe_keywords([question])


def _apply_recommendation_tag_rhythm(
    *,
    content_tags: Sequence[str],
    question: str,
    scene: str,
    trigger_type: str,
    history: Sequence[ChatMessageRow],
    route_kind: str,
    focus_node: Optional[dict[str, Any]],
    toc_nodes: Sequence[dict[str, Any]],
) -> _TagRhythmResult:
    turn_index = _v5_turn_index(history)
    effective_scene = scene
    if route_kind == "explore_product":
        explored = explore_product_title_from_tag(question) or str(focus_node.get("title") if focus_node else "")
        mapped_scene = scene_for_toc_title(explored)
        if mapped_scene:
            effective_scene = mapped_scene
    tags = fallback_tags_for_scene(effective_scene)
    seen_guide = route_kind == "guide" or _history_has_tag_kind(history=history, scene=effective_scene, kind="guide")
    seen_case = route_kind == "case" or _history_has_tag_kind(history=history, scene=effective_scene, kind="case")
    stage = "fixed_entry"

    if seen_guide:
        tags[0] = price_tag_for_scene(effective_scene)
        stage = "guide_to_price"

    used_explore_products = False
    # 仅在「刚看完优秀案例库」这一轮展示其他产品的探索标签；后续恢复常规节奏
    if route_kind == "case":
        explore_tags = _platform_intro_explore_tags(scene=effective_scene, toc_nodes=toc_nodes, limit=3)
        if explore_tags:
            tags = explore_tags
            used_explore_products = True
            stage = "case_to_explore_products"
        else:
            tags[0] = price_tag_for_scene(effective_scene)
            stage = "case_to_price"

    if not used_explore_products:
        tags[2] = trial_tag_for_scene(effective_scene)
    tags = _dedupe_tag_list(tags)[:3]
    if len(tags) < 3 and not used_explore_products:
        for fallback in fallback_tags_for_scene(effective_scene):
            if len(tags) >= 3:
                break
            if _norm_for_match(fallback) not in {_norm_for_match(tag) for tag in tags}:
                tags.append(fallback)

    return _TagRhythmResult(
        tags=tags,
        conversion_state={
            "turn_index": turn_index,
            "stage": stage,
            "trigger_type": trigger_type,
            "tag_route_kind": route_kind,
            "seen_guide": seen_guide,
            "seen_case": seen_case,
        },
    )


def _v5_turn_index(history: Sequence[ChatMessageRow]) -> int:
    user_turns = sum(1 for row in history if getattr(row, "role", "") == "user")
    return max(1, user_turns + 1)


def _clean_content_tags(tags: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for raw in tags:
        tag = str(raw or "").strip()
        norm = _norm_for_match(tag)
        if not tag or not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(tag[:60])
        if len(out) >= 3:
            break
    return out


def _dedupe_tag_list(tags: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for raw in tags:
        tag = str(raw or "").strip()
        norm = _norm_for_match(tag)
        if not tag or not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(tag[:60])
    return out


def _case_query_context(*, question: str, scene: str, history: Sequence[ChatMessageRow]) -> str:
    parts: List[str] = [scene]
    for row in list(history)[-8:]:
        content = (getattr(row, "content", "") or "").strip()
        if not content or classify_friend_v5_tag(content, scene=scene) == "case":
            continue
        if getattr(row, "role", "") == "user":
            parts.append(content)
    parts.append(question)
    return " ".join(parts).strip()


def _should_suppress_initial_media(*, trigger_type: str, tag_route: _TagRouteResult) -> bool:
    return trigger_type == "scene" or tag_route.kind == "explore_product"


async def _load_admin_scene_media(
    *,
    repository: Optional[Any],
    scene: str,
    trigger_type: str,
    history: Sequence[ChatMessageRow],
) -> ChatMediaBundle:
    if repository is None or trigger_type != "scene" or history:
        return ChatMediaBundle()
    scene_key = _admin_scene_key(scene)
    if not scene_key:
        return ChatMediaBundle()
    try:
        rows = await repository.list_videos(scene_key=scene_key)
    except Exception:
        logger.exception("friend_v5_admin_scene_video_load_failed")
        return ChatMediaBundle()
    videos = [
        MediaItem(
            url=str(getattr(row, "file_url", "") or ""),
            title=str(getattr(row, "title", "") or getattr(row, "original_filename", "") or ""),
            doc_title=str(getattr(row, "scene_name", "") or scene),
            doc_id=f"admin_video:{getattr(row, 'id', '')}",
        )
        for row in rows[:1]
        if str(getattr(row, "file_url", "") or "").strip()
    ]
    return ChatMediaBundle(videos=videos)


def _admin_scene_key(scene: str) -> str:
    normalized = (scene or "").strip()
    return {
        "人工智能通识教育": "general_ai_course",
        "人工智能通识课程": "general_ai_course",
        "跨学科项目化学习": "project_based_learning",
        "跨学科项目式学习": "project_based_learning",
        "智能招生": "smart_enrollment",
        "学校AI场景定制": "school_ai_custom",
    }.get(normalized, "")


def _merge_media(primary: ChatMediaBundle, extra: ChatMediaBundle) -> ChatMediaBundle:
    if not extra.images and not extra.videos:
        return primary
    image_urls = {item.url for item in primary.images}
    extra_video_urls = {item.url for item in extra.videos}
    return ChatMediaBundle(
        images=[*primary.images, *[item for item in extra.images if item.url not in image_urls]],
        videos=[*extra.videos, *[item for item in primary.videos if item.url not in extra_video_urls]],
    )


def _resolve_tag_route(
    *,
    question: str,
    scene: str,
    trigger_type: str,
    history: Sequence[ChatMessageRow],
    toc_nodes: Sequence[dict[str, Any]],
) -> _TagRouteResult:
    if trigger_type != "tag":
        return _TagRouteResult(kind="none")
    kind = classify_friend_v5_tag(question, scene=scene)
    explicit_product = try_product_title_from_tag(question)
    target_title = product_title_from_tag(question, scene=scene)
    if kind == "guide":
        focus = _focus_under_path(["使用指南", target_title], toc_nodes)
        return _TagRouteResult(kind="guide", target_title=target_title, focus_node=focus)
    if kind == "case":
        if explicit_product:
            focus = _focus_under_path(
                [_CASE_SECTION_TITLE, _CASE_LIBRARY_TITLE, explicit_product],
                toc_nodes,
            )
            return _TagRouteResult(kind="case", target_title=explicit_product, focus_node=focus)
        focus = _best_case_toc_match(
            question=question,
            scene=scene,
            history=history,
            toc_nodes=toc_nodes,
        )
        return _TagRouteResult(kind="case", target_title=target_title, focus_node=focus)
    if kind == "price":
        return _TagRouteResult(kind="price", target_title=target_title)
    if kind == "trial":
        return _TagRouteResult(kind="trial", target_title=target_title)
    explore_title = explore_product_title_from_tag(question)
    if explore_title:
        focus = _resolve_platform_intro_focus(product_title=explore_title, toc_nodes=toc_nodes)
        return _TagRouteResult(kind="explore_product", target_title=explore_title, focus_node=focus)
    focus = _best_toc_match(question=question, scene="", parsed_tags=[], toc_nodes=toc_nodes)
    return _TagRouteResult(kind="toc", target_title=(question or "").strip(), focus_node=focus)


def _focus_under_path(path_titles: Sequence[str], toc_nodes: Sequence[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not path_titles or not toc_nodes:
        return None
    wanted = [_norm_for_match(title) for title in path_titles if str(title or "").strip()]
    if not wanted:
        return None
    fallback_parent: Optional[dict[str, Any]] = None
    for node in toc_nodes:
        path = [_norm_for_match(title) for title in (node.get("path") or [])]
        if len(path) >= len(wanted) and path[-len(wanted) :] == wanted:
            if str(node.get("doc_id") or "").strip():
                return node
            fallback_parent = fallback_parent or node
    if fallback_parent:
        return _first_readable_descendant(fallback_parent, toc_nodes) or fallback_parent
    return None


def _history_has_tag_kind(*, history: Sequence[ChatMessageRow], scene: str, kind: str) -> bool:
    for row in history:
        if getattr(row, "role", "") != "user":
            continue
        if classify_friend_v5_tag(str(getattr(row, "content", "") or ""), scene=scene) == kind:
            return True
    return False


def _history_has_seen_case(history: Sequence[ChatMessageRow]) -> bool:
    return _history_active_content_branch(history) == "case"


def _history_has_seen_guide(history: Sequence[ChatMessageRow]) -> bool:
    return _history_active_content_branch(history) == "guide"


def _history_active_content_branch(history: Sequence[ChatMessageRow]) -> Optional[str]:
    active: Optional[str] = None
    for row in history:
        if getattr(row, "role", "") != "user":
            continue
        content = str(getattr(row, "content", "") or "")
        kind = classify_friend_v5_tag(content, scene="")
        if kind in {"case", "guide"}:
            active = kind
    return active


def _resolve_guide_focus(
    *,
    product_title: str,
    toc_nodes: Sequence[dict[str, Any]],
) -> tuple[Optional[dict[str, Any]], str, bool]:
    title = (product_title or "").strip()
    if not title or not toc_nodes:
        return None, title, False
    focus = _focus_under_path(["使用指南", title], toc_nodes)
    if focus is None:
        return None, title, False
    return focus, title, True


def _resolve_platform_intro_focus(
    *,
    product_title: str,
    toc_nodes: Sequence[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    title = (product_title or "").strip()
    if not title or not toc_nodes:
        return None
    return _focus_under_path([_PLATFORM_SECTION_TITLE, title], toc_nodes)


async def _persist_contact_from_question(
    *,
    profile_repo: Any,
    session_id: str,
    question: str,
    profile: Any,
    scene: str,
) -> Any:
    contact_hit = extract_contact(question)
    if not contact_hit:
        return profile
    interests = dict(profile.interests) if profile and isinstance(getattr(profile, "interests", None), dict) else {}
    lead = dict(interests.get("_lead") or {})
    lead.update(
        {
            "contact_type": contact_hit.contact_type,
            "contact_value": contact_hit.value,
        }
    )
    if not str(lead.get("interested_product") or "").strip():
        lead["interested_product"] = toc_title_for_scene(scene)
    interests["_lead"] = lead
    await profile_repo.upsert_profile(session_id=session_id, interests=interests)
    return await profile_repo.get_profile(session_id=session_id)


def _platform_intro_explore_tags(
    *,
    scene: str,
    toc_nodes: Sequence[dict[str, Any]],
    limit: int = 3,
) -> List[str]:
    if not toc_nodes:
        return []
    current_title = toc_title_for_scene(scene)
    current_norm = _norm_for_match(current_title)
    platform_uuid = ""
    for node in toc_nodes:
        if _norm_for_match(str(node.get("title") or "")) == _norm_for_match(_PLATFORM_SECTION_TITLE):
            platform_uuid = str(node.get("uuid") or "")
            break
    if not platform_uuid:
        return []
    tags: List[str] = []
    for node in toc_nodes:
        if str(node.get("parent_uuid") or "") != platform_uuid:
            continue
        title = str(node.get("title") or "").strip()
        if not title or _norm_for_match(title) == current_norm:
            continue
        tag = explore_product_tag_for_title(title)
        if tag and tag not in tags:
            tags.append(tag)
        if len(tags) >= limit:
            break
    return tags


def _scene_mcp_mode(*, scene_case_continuation: bool, scene_guide_continuation: bool) -> str:
    if scene_case_continuation:
        return "scene_case_library"
    if scene_guide_continuation:
        return "scene_guide"
    return "scene_toc"


def _is_bare_product_question(question: str, *, scene: str) -> bool:
    raw = (question or "").strip()
    if not raw:
        return False
    if classify_friend_v5_tag(raw, scene=scene) in {"guide", "case", "price", "trial"}:
        return False
    product = _product_title_from_question(raw, scene=scene)
    if not product:
        return False
    product_norm = _norm_for_match(product)
    raw_norm = _norm_for_match(raw)
    if raw_norm == product_norm:
        return True
    for scene_key, toc_title in SCENE_TO_TOC_TITLE.items():
        if raw_norm in {_norm_for_match(scene_key), _norm_for_match(toc_title)}:
            return True
    return False


def _case_focused_prompt_block(prompt_block: str) -> str:
    block = (prompt_block or "").strip()
    if not block:
        return block
    excerpt = block
    for marker in ("解读：", "解读:", "IDEAS-PBL", "IDEAS PBL", "智能设计与评价系统"):
        idx = excerpt.find(marker)
        if idx > 400:
            excerpt = excerpt[:idx].rstrip()
            break
    if excerpt != block:
        excerpt += "\n（后续为产品解读段落，本轮优先讲落地学校案例，除非用户追问系统能力。）"
    return excerpt


def _product_title_from_question(question: str, *, scene: str) -> str:
    raw = (question or "").strip()
    if not raw:
        return ""
    raw_norm = _norm_for_match(raw)
    for scene_key, toc_title in SCENE_TO_TOC_TITLE.items():
        if raw_norm in {_norm_for_match(scene_key), _norm_for_match(toc_title)}:
            return toc_title
    mapped = toc_title_for_scene(scene)
    if mapped and raw_norm == _norm_for_match(scene):
        return mapped
    return raw


def _resolve_case_library_focus(
    *,
    product_title: str,
    scene: str,
    history: Sequence[ChatMessageRow],
    toc_nodes: Sequence[dict[str, Any]],
) -> tuple[Optional[dict[str, Any]], str, bool]:
    title = (product_title or "").strip()
    if not title or not toc_nodes:
        return None, title, False
    focus = _focus_under_path([_CASE_SECTION_TITLE, _CASE_LIBRARY_TITLE, title], toc_nodes)
    if focus is None:
        focus = _focus_under_path([_CASE_LIBRARY_TITLE, title], toc_nodes)
    if focus is None:
        focus = _best_case_toc_match(
            question=title,
            scene=title,
            history=history,
            toc_nodes=toc_nodes,
        )
    if focus is None:
        return None, title, False
    return focus, title, True


def _topic_already_discussed(title: str, discussed: str) -> bool:
    title_norm = _norm_for_match(title)
    discussed_norm = _norm_for_match(discussed)
    if not title_norm:
        return False
    if title_norm in discussed_norm:
        return True
    for token in _match_tokens(title_norm):
        if len(token) >= 3 and token in discussed_norm:
            return True
    return False


def _first_case_sibling_tag(
    *,
    focus_node: Optional[dict[str, Any]],
    toc_nodes: Sequence[dict[str, Any]],
) -> Optional[str]:
    if not focus_node:
        return None
    by_uuid = {str(node.get("uuid") or ""): node for node in toc_nodes if str(node.get("uuid") or "")}
    for sibling_uuid in focus_node.get("siblings") or []:
        sibling = by_uuid.get(str(sibling_uuid or ""))
        title = str((sibling or {}).get("title") or "").strip()
        if title:
            return title[:60]
    return None


def _first_content_tag(content_tags: Sequence[str], *, blocked: Sequence[str]) -> Optional[str]:
    blocked_norms = {_norm_for_match(tag) for tag in blocked}
    for raw in content_tags:
        tag = str(raw or "").strip()
        norm = _norm_for_match(tag)
        if tag and norm and norm not in blocked_norms:
            return tag[:60]
    return None


def _followup_topic_for_question(
    *,
    question: str,
    trigger_type: str,
    history: Sequence[ChatMessageRow],
) -> str:
    if trigger_type != "manual" or not _FOLLOWUP_CONFIRM_RE.search((question or "").strip()):
        return ""
    for row in reversed(history):
        if getattr(row, "role", "") != "assistant":
            continue
        content = str(getattr(row, "content", "") or "")
        match = _FOLLOWUP_TOPIC_RE.search(content)
        if match:
            return match.group(1).strip(" ：:，,。?")
    return ""


def _select_next_followup_topic(
    *,
    focus: Optional[dict[str, Any]],
    toc_nodes: Sequence[dict[str, Any]],
    deep_read: FriendV5YuqueDeepReadResult,
    answer: str,
    history: Sequence[ChatMessageRow] = (),
) -> str:
    by_uuid = {str(node.get("uuid") or ""): node for node in toc_nodes if str(node.get("uuid") or "")}
    discussed = _history_text(history)
    if focus:
        for child_uuid in focus.get("children") or []:
            child = by_uuid.get(str(child_uuid or ""))
            title = str((child or {}).get("title") or "").strip()
            if title and not _topic_already_discussed(title, discussed):
                return title[:40]
    title = _topic_from_deep_read(deep_read) or _topic_from_answer(answer)
    return title[:40] if title else ""


def _select_followup_sibling_topic(
    *,
    focus: Optional[dict[str, Any]],
    toc_nodes: Sequence[dict[str, Any]],
    history: Sequence[ChatMessageRow],
    exclude: Sequence[str],
) -> str:
    """同级目录里挑一个未介绍过的话题，用于文末横向引导（仅限同一产品下的子话题）。"""
    if not focus:
        return ""
    by_uuid = {str(node.get("uuid") or ""): node for node in toc_nodes if str(node.get("uuid") or "")}
    discussed = _history_text(history)
    exclude_norms = {_norm_for_match(item) for item in exclude if str(item or "").strip()}

    child_candidates: List[str] = []
    for child_uuid in focus.get("children") or []:
        child = by_uuid.get(str(child_uuid or ""))
        title = str((child or {}).get("title") or "").strip()
        if not title or _norm_for_match(title) in exclude_norms or _topic_already_discussed(title, discussed):
            continue
        child_candidates.append(title[:40])
    if child_candidates:
        return child_candidates[0]

    parent_uuid = str(focus.get("parent_uuid") or "")
    parent = by_uuid.get(parent_uuid) if parent_uuid else None
    parent_title = str((parent or {}).get("title") or "").strip()
    if parent_title and _norm_for_match(parent_title) in {
        _norm_for_match("平台介绍"),
        _norm_for_match("使用指南"),
        _norm_for_match(_CASE_SECTION_TITLE),
        _norm_for_match(_CASE_LIBRARY_TITLE),
    }:
        return ""

    for sibling_uuid in focus.get("siblings") or []:
        sibling = by_uuid.get(str(sibling_uuid or ""))
        title = str((sibling or {}).get("title") or "").strip()
        if not title or _norm_for_match(title) in exclude_norms or _topic_already_discussed(title, discussed):
            continue
        return title[:40]
    return ""


def _history_text(history: Sequence[ChatMessageRow]) -> str:
    return "".join(str(getattr(row, "content", "") or "") for row in history)


def _topic_from_deep_read(deep_read: FriendV5YuqueDeepReadResult) -> str:
    prompt = str(getattr(deep_read, "prompt_block", "") or "")
    for pattern in (r"标题[:：]\s*([^\n]{2,40})", r"正文摘录[:：]\s*([^。\n]{4,40})"):
        match = re.search(pattern, prompt)
        if match:
            return match.group(1).strip()
    return ""


def _topic_from_answer(answer: str) -> str:
    text = re.sub(r"\s+", "", answer or "")
    for pattern in (r"(课堂流程[^。！？]{0,20})", r"(课程目标[^。！？]{0,20})", r"(落地流程[^。！？]{0,20})", r"(使用步骤[^。！？]{0,20})"):
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return ""


def _strip_existing_followup_questions(text: str) -> str:
    lines: List[str] = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        if _FOLLOWUP_TOPIC_RE.search(stripped) or "也感兴趣，也可以为您介绍" in stripped:
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _append_followup_question(answer: str, topic: str, *, sibling_topic: str = "") -> str:
    text = (answer or "").strip()
    clean_topic = (topic or "").strip()
    if not text or not clean_topic:
        return text
    line = f"需要我和你详细介绍{clean_topic}的内容吗？"
    clean_sibling = (sibling_topic or "").strip()
    if clean_sibling and clean_sibling != clean_topic:
        line += f"如果您对{clean_sibling}也感兴趣，也可以为您介绍。"
    return f"{text}\n\n{line}"


def _strip_inline_urls(text: str) -> str:
    """正文不直接展示链接：Markdown 链接保留标题文本，裸链接（含 www. 开头）整体移除。"""
    out = re.sub(r"\[([^\]]+)\]\((?:https?://|www\.)[^)]+\)", r"\1", text or "")
    out = re.sub(r"(?:https?://|www\.)[^\s)\]}>\"'，。、；;：]+", "", out)
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _product_matches_scene(question: str, *, scene: str) -> bool:
    product = _product_title_from_question(question, scene=scene)
    scene_product = toc_title_for_scene(scene)
    if not product or not scene_product:
        return False
    return _norm_for_match(product) == _norm_for_match(scene_product)


def _is_other_scene_root_product(question: str, *, scene: str) -> bool:
    raw = (question or "").strip()
    if not raw:
        return False
    product = _product_title_from_question(raw, scene=scene)
    product_norm = _norm_for_match(product)
    raw_norm = _norm_for_match(raw)
    current_norms = {_norm_for_match(scene), _norm_for_match(toc_title_for_scene(scene))}
    if product_norm in current_norms or raw_norm in current_norms:
        return False
    for other_scene, toc_title in SCENE_TO_TOC_TITLE.items():
        if other_scene == scene:
            continue
        other_norms = {_norm_for_match(other_scene), _norm_for_match(toc_title)}
        if product_norm in other_norms or raw_norm in other_norms:
            return True
    return False


def _cross_scene_redirect_answer(*, scene: str, product: str) -> str:
    clean_product = (product or "").strip() or "该方向"
    return (
        f"您当前在了解「{scene}」。如果想了解「{clean_product}」，"
        f"请先在左侧点击对应场景，我再为您介绍该方向的平台介绍、使用指南和优秀案例。"
    )


def _append_guide_inline_link(
    answer: str,
    sources: Sequence[FriendV5SourceItem],
    *,
    scene: str,
) -> str:
    text = (answer or "").strip()
    yuque = next(
        (item for item in sources if item.source_type == "yuque" and (item.url or "").strip()),
        None,
    )
    title = toc_title_for_scene(scene) or (yuque.title if yuque else "")
    if yuque and yuque.url:
        link_line = f"详细操作请点击：[{title}使用指南]({yuque.url})"
    else:
        link_line = _GUIDE_SOURCES_HINT
    if _GUIDE_SOURCES_HINT in text:
        return text.replace(_GUIDE_SOURCES_HINT, link_line)
    if link_line in text:
        return text
    return f"{text}\n\n{link_line}"


def _case_kb_fallback_answer() -> str:
    return _CASE_KB_FALLBACK_ANSWER


def _price_handoff_answer(*, scene: str, profile: Any) -> str:
    title = toc_title_for_scene(scene)
    name = str(getattr(profile, "display_name", "") or "").strip()
    org = str(getattr(profile, "org_name", "") or "").strip()
    prefix_parts = [part for part in (name, org) if part]
    prefix = f"{'，'.join(prefix_parts)}，" if prefix_parts else ""
    body = (
        f"{prefix}{title}的价格会根据学校规模、使用场景、账号数量和服务支持范围来确定，"
        "通常需要先做一次简单沟通，才能给到更准确的方案区间。\n\n"
    )
    if name:
        ask = "我可以先帮您登记，后续安排顾问对接详细报价。方便留个微信或电话吗？"
    elif org:
        ask = "我可以先帮您登记，后续安排顾问对接详细报价。方便告诉我怎么称呼您，并留个微信或电话吗？"
    else:
        ask = "我可以先帮您登记，后续安排顾问对接详细报价。方便留下您的称呼和联系方式吗？"
    return body + ask


def _tag_route_debug(route: _TagRouteResult) -> dict[str, Any]:
    return {
        "kind": route.kind,
        "target_title": route.target_title,
        "focus_node": _catalog_focus_debug(route.focus_node),
    }


def _skill_route_debug(route: Optional[SkillRoute]) -> Optional[dict[str, Any]]:
    if not route:
        return None
    return {"skill_id": route.skill_id}


def _mcp_route_debug(mode: str, query: str, focus: Optional[dict[str, Any]]) -> dict[str, Any]:
    return {
        "mode": mode,
        "query": (query or "").strip(),
        "focus_node": _catalog_focus_debug(focus),
    }


def _get(raw: Any, key: str) -> Any:
    if isinstance(raw, dict):
        return raw.get(key)
    return None


def _normalize_toc_nodes(raw_nodes: Sequence[dict[str, Any]]) -> List[dict[str, Any]]:
    nodes: List[dict[str, Any]] = []
    seen: set[str] = set()
    for idx, raw in enumerate(raw_nodes):
        title = str(_get(raw, "title") or "").strip()
        if not title:
            continue
        uuid = str(_get(raw, "uuid") or f"toc-{idx}").strip()
        key = uuid or title
        if key in seen:
            continue
        seen.add(key)
        try:
            level = max(1, int(_get(raw, "level") or 1))
        except Exception:
            level = 1
        nodes.append(
            {
                "uuid": uuid,
                "title": title,
                "level": level,
                "parent_uuid": str(_get(raw, "parent_uuid") or "").strip(),
                "node_type": str(_get(raw, "node_type") or "").strip(),
                "url": _get(raw, "url"),
                "doc_id": _get(raw, "doc_id"),
            }
        )
    _attach_toc_navigation(nodes)
    return nodes


def _catalog_matched_tags(
    *,
    question: str,
    scene: str,
    parsed_tags: Sequence[str],
    toc_nodes: Sequence[dict[str, Any]],
    focus: Optional[dict[str, Any]] = None,
) -> _CatalogTagResult:
    if not toc_nodes:
        return _CatalogTagResult(tags=list(parsed_tags)[:3], focus_node=None)

    by_uuid = {str(node.get("uuid") or ""): node for node in toc_nodes if str(node.get("uuid") or "")}
    children: dict[str, List[dict[str, Any]]] = {}
    for node in toc_nodes:
        parent = str(node.get("parent_uuid") or "")
        if parent:
            children.setdefault(parent, []).append(node)

    focus = focus or _best_toc_match(question=question, scene=scene, parsed_tags=parsed_tags, toc_nodes=toc_nodes)
    ordered: List[dict[str, Any]] = []
    if focus:
        focus_uuid = str(focus.get("uuid") or "")
        focus_children = children.get(focus_uuid, [])
        parent_uuid = str(focus.get("parent_uuid") or "")
        if parent_uuid:
            siblings = [node for node in children.get(parent_uuid, []) if node is not focus]
        else:
            siblings = [node for node in toc_nodes if not str(node.get("parent_uuid") or "") and node is not focus]
        if focus_children:
            ordered.extend(siblings[:1])
            ordered.extend(focus_children[:2])
            ordered.extend(focus_children[2:])
            ordered.extend(siblings[1:])
        else:
            ordered.extend(siblings[:3])
        if parent_uuid:
            parent = by_uuid.get(parent_uuid)
            if parent:
                grandparent_uuid = str(parent.get("parent_uuid") or "")
                if grandparent_uuid:
                    ordered.extend([node for node in children.get(grandparent_uuid, []) if node is not parent])

    tags: List[str] = []
    seen_titles: set[str] = set()
    for node in ordered:
        title = str(node.get("title") or "").strip()
        if not title or title in seen_titles:
            continue
        if title == scene and len(toc_nodes) > 1:
            continue
        seen_titles.add(title)
        tags.append(title[:60])
        if len(tags) >= 3:
            break
    return _CatalogTagResult(tags=tags, focus_node=focus)


def _catalog_focus_debug(node: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not node:
        return None
    return {
        "uuid": str(node.get("uuid") or ""),
        "title": str(node.get("title") or ""),
        "path": list(node.get("path") or []),
    }


def _best_case_toc_match(
    *,
    question: str,
    scene: str,
    history: Sequence[ChatMessageRow],
    toc_nodes: Sequence[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    if not toc_nodes:
        return None
    children = _children_by_parent(toc_nodes)
    library = _case_library_node(toc_nodes)
    if not library:
        return None
    descendants = _descendants(library, children)
    candidates = [
        node
        for node in descendants
        if str(node.get("title") or "").strip() and str(node.get("node_type") or "").strip() != "title"
    ]
    if not candidates:
        candidates = [node for node in descendants if str(node.get("title") or "").strip()]
    if not candidates:
        return library

    context = _case_query_context(question=question, scene=scene, history=history)
    context_norm = _norm_for_match(context)
    context_tokens = set(_match_tokens(context_norm))
    scored: List[tuple[int, int, dict[str, Any]]] = []
    representative: Optional[dict[str, Any]] = None
    for idx, node in enumerate(candidates):
        title = str(node.get("title") or "").strip()
        title_norm = _norm_for_match(title)
        if not title_norm:
            continue
        if "代表" in title or "通用" in title:
            representative = representative or node
        score = 0
        if title_norm and title_norm in context_norm:
            score += 1000 + len(title_norm)
        token_score = sum(len(token) for token in set(_match_tokens(title_norm)).intersection(context_tokens))
        if token_score:
            score += 100 + token_score
        if scene and _norm_for_match(scene) in title_norm:
            score += 80
        scored.append((score, -idx, node))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    if scored and scored[0][0] > 0:
        return scored[0][2]
    return representative or candidates[0]


def _case_library_node(toc_nodes: Sequence[dict[str, Any]]) -> Optional[dict[str, Any]]:
    by_uuid = {str(node.get("uuid") or ""): node for node in toc_nodes if str(node.get("uuid") or "")}
    case_section_norm = _norm_for_match(_CASE_SECTION_TITLE)
    library_norm = _norm_for_match(_CASE_LIBRARY_TITLE)
    fallback_library: Optional[dict[str, Any]] = None
    fallback_section: Optional[dict[str, Any]] = None
    for node in toc_nodes:
        title_norm = _norm_for_match(str(node.get("title") or ""))
        if title_norm == case_section_norm:
            fallback_section = fallback_section or node
        if title_norm != library_norm:
            continue
        fallback_library = fallback_library or node
        if case_section_norm in [_norm_for_match(title) for title in _toc_path_titles(node, by_uuid)]:
            return node
    return fallback_library or fallback_section


async def _read_deep_by_focus_or_query(
    *,
    reader: Any,
    focus: Optional[dict[str, Any]],
    question: str,
    allow_search_fallback: bool = True,
) -> FriendV5YuqueDeepReadResult:
    if focus and hasattr(reader, "read_toc_node"):
        result = await reader.read_toc_node(node=focus, question=question)
        if getattr(result, "used", False):
            return result
        if not allow_search_fallback:
            return result
    if allow_search_fallback:
        return await reader.read(question=question)
    return FriendV5YuqueDeepReadResult(debug={"mode": "toc_focus_read_miss"})


def _attach_toc_navigation(nodes: List[dict[str, Any]]) -> None:
    by_uuid = {str(node.get("uuid") or ""): node for node in nodes if str(node.get("uuid") or "")}
    children = _children_by_parent(nodes)
    for node in nodes:
        uuid = str(node.get("uuid") or "")
        parent_uuid = str(node.get("parent_uuid") or "")
        node["children"] = [str(child.get("uuid") or "") for child in children.get(uuid, [])]
        node["siblings"] = [
            str(sibling.get("uuid") or "")
            for sibling in children.get(parent_uuid, [])
            if sibling is not node
        ]
        node["path"] = _toc_path_titles(node, by_uuid)


def _children_by_parent(nodes: Sequence[dict[str, Any]]) -> dict[str, List[dict[str, Any]]]:
    children: dict[str, List[dict[str, Any]]] = {}
    for node in nodes:
        children.setdefault(str(node.get("parent_uuid") or ""), []).append(node)
    return children


def _toc_path_titles(node: dict[str, Any], by_uuid: dict[str, dict[str, Any]]) -> List[str]:
    path: List[str] = []
    seen: set[str] = set()
    current: Optional[dict[str, Any]] = node
    while current:
        title = str(current.get("title") or "").strip()
        if title:
            path.append(title)
        parent_uuid = str(current.get("parent_uuid") or "")
        if not parent_uuid or parent_uuid in seen:
            break
        seen.add(parent_uuid)
        current = by_uuid.get(parent_uuid)
    return list(reversed(path))


def _focus_by_mapped_tag(
    *,
    question: str,
    toc_nodes: Sequence[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    """兼容旧调用：按标题直接定位语雀目录；目录分组无 doc_id 时下钻到首个可读子文档。"""
    title = (question or "").strip()
    if not title or not toc_nodes:
        return None
    target_norm = _norm_for_match(title)
    matched: Optional[dict[str, Any]] = None
    for node in toc_nodes:
        if _norm_for_match(str(node.get("title") or "")) == target_norm:
            matched = node
            break
    if matched is None:
        return None
    if str(matched.get("doc_id") or "").strip():
        return matched
    # 分组标题（TITLE）没有 doc_id，下钻到第一个带 doc_id 的子文档
    child = _first_readable_descendant(matched, toc_nodes)
    return child or matched


def _first_readable_descendant(
    node: dict[str, Any],
    toc_nodes: Sequence[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    by_uuid = {str(n.get("uuid") or ""): n for n in toc_nodes if str(n.get("uuid") or "")}
    stack: List[str] = [str(uid) for uid in (node.get("children") or [])]
    seen: set[str] = set()
    while stack:
        uid = stack.pop(0)
        if uid in seen:
            continue
        seen.add(uid)
        child = by_uuid.get(uid)
        if child is None:
            continue
        if str(child.get("doc_id") or "").strip():
            return child
        stack[0:0] = [str(c) for c in (child.get("children") or [])]
    return None


def _best_toc_match(
    *,
    question: str,
    scene: str,
    parsed_tags: Sequence[str],
    toc_nodes: Sequence[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    probes = [question, scene, *parsed_tags]
    best: Optional[dict[str, Any]] = None
    best_score = 0
    for node in toc_nodes:
        title = str(node.get("title") or "").strip()
        if not title:
            continue
        title_norm = _norm_for_match(title)
        score = 0
        for probe in probes:
            probe_norm = _norm_for_match(probe)
            if not probe_norm:
                continue
            if title_norm == probe_norm:
                score = max(score, 1000 + len(title_norm))
            elif title_norm in probe_norm:
                score = max(score, 700 + len(title_norm))
            elif probe_norm in title_norm:
                score = max(score, 500 + len(probe_norm))
            else:
                title_tokens = set(_match_tokens(title_norm))
                probe_tokens = set(_match_tokens(probe_norm))
                token_score = sum(len(token) for token in title_tokens.intersection(probe_tokens))
                if token_score >= 6:
                    score = max(score, 100 + token_score)
        if score > best_score:
            best = node
            best_score = score
    return best


def _rank_toc_nodes(
    *,
    question: str,
    scene: str,
    parsed_tags: Sequence[str],
    toc_nodes: Sequence[dict[str, Any]],
) -> List[dict[str, Any]]:
    probes = [_norm_for_match(question), _norm_for_match(scene), *[_norm_for_match(tag) for tag in parsed_tags]]
    scored: List[tuple[int, int, dict[str, Any]]] = []
    for idx, node in enumerate(toc_nodes):
        title = str(node.get("title") or "")
        title_norm = _norm_for_match(title)
        if not title_norm:
            continue
        score = 0
        for probe in probes:
            if not probe:
                continue
            if title_norm in probe or probe in title_norm:
                score += min(len(title_norm), len(probe))
            for token in _match_tokens(probe):
                if token and token in title_norm:
                    score += len(token)
        try:
            level = int(node.get("level") or 1)
        except Exception:
            level = 1
        scored.append((score, -level, node if score > 0 else {**node, "_idx": idx}))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in scored]


def _descendants(node: dict[str, Any], children: dict[str, List[dict[str, Any]]]) -> List[dict[str, Any]]:
    out: List[dict[str, Any]] = []
    stack = list(children.get(str(node.get("uuid") or ""), []))
    while stack:
        current = stack.pop(0)
        out.append(current)
        stack[0:0] = children.get(str(current.get("uuid") or ""), [])
    return out


def _norm_for_match(value: str) -> str:
    return re.sub(r"[\s「」『』《》【】\[\]（）()、，。:：;；?？!！\-_/|]+", "", str(value or "").lower())


def _match_tokens(value: str) -> List[str]:
    text = _norm_for_match(value)
    tokens = [t for t in re.split(r"(ai|人工智能|智能|招生|课程|项目|学校|校本|实验室|教师|培训|家长|课堂|案例|方案)", text) if len(t) >= 2]
    return tokens
