from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, AsyncIterator, List, Optional, Sequence

from app.conversation.friend_persona_v5 import build_friend_v5_system_prompt, scene_description
from app.conversation.profile_extractor import ProfileExtractor
from app.core.logger import get_logger
from app.db.repositories import ChatMessageRow

logger = get_logger(__name__)
from app.rag.friend_v5_generator import FriendV5Generator
from app.schemas.chat_v5 import ChatV5DonePayload, FriendV5SourceItem
from app.service.friend_v5_tags import (
    TAG_TO_TOC_TITLE,
    FriendV5TagStreamFilter,
    fallback_tags_for_scene,
)
from app.service.friend_v5_yuque_deep_reader import (
    FriendV5YuqueDeepReadResult,
    should_deep_read_yuque_doc,
)


_MANUAL_YUQUE_HINT_RE = re.compile(
    r"(课程|产品|方案|指南|文档|手册|介绍|案例|乐高|AI|人工智能|项目化|招生|实验室|校本)"
)
_PRODUCT_CASE_TAG_NORMS = {"产品案例", "案例"}
_PRODUCT_CASE_TAG = "产品案例"
_PRICE_TAG = "产品价格"
_TRIAL_TAG = "如何申请内测"
_CONVERSION_TAGS = {_PRICE_TAG, _TRIAL_TAG}
_CASE_SECTION_TITLE = "案例与社区"
_CASE_LIBRARY_TITLE = "优秀案例库"


@dataclass(frozen=True)
class _CatalogTagResult:
    tags: List[str]
    focus_node: Optional[dict[str, Any]]


@dataclass(frozen=True)
class _TagRhythmResult:
    tags: List[str]
    conversion_state: dict[str, Any]


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
        if update.display_name or update.org_name or update.interests:
            await self._profile_repo.upsert_profile(
                session_id=session_id,
                display_name=update.display_name,
                org_name=update.org_name,
                interests=update.interests,
            )
            profile = await self._profile_repo.get_profile(session_id=session_id)

        deep_read = FriendV5YuqueDeepReadResult()
        scene_query_rewrite: dict[str, Any] = {}
        scene_query = (question or "").strip()
        catalog_scene = scene if trigger_type == "scene" else ""
        catalog_focus = _best_toc_match(question=question, scene=catalog_scene, parsed_tags=[], toc_nodes=self._toc_nodes)
        # 兜底标签（如「想看看使用指南？」）点击后，优先定位到映射的语雀目录
        mapped_focus = _focus_by_mapped_tag(question=question, toc_nodes=self._toc_nodes)
        if mapped_focus is not None:
            catalog_focus = mapped_focus
        case_branch_used = False
        if trigger_type == "scene" and self._scene_query_rewriter:
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

        if trigger_type == "scene" and self._yuque_deep_reader and scene_query:
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
            and _is_product_case_request(question=question, trigger_type=trigger_type)
            and self._yuque_deep_reader
        ):
            case_branch_used = True
            case_focus = _best_case_toc_match(
                question=question,
                scene=scene,
                history=history,
                toc_nodes=self._toc_nodes,
            )
            if case_focus:
                catalog_focus = case_focus
                yield _stage("yuque_case_read", "小为正在匹配案例库里的真实案例...")
                try:
                    deep_read = await _read_deep_by_focus_or_query(
                        reader=self._yuque_deep_reader,
                        focus=case_focus,
                        question=_case_query_context(question=question, scene=scene, history=history),
                    )
                except Exception:
                    deep_read = FriendV5YuqueDeepReadResult(debug={"mode": "case_deep_read_error"})

        if (
            not deep_read.used
            and trigger_type == "tag"
            and self._yuque_deep_reader
            and catalog_focus
        ):
            yield _stage("yuque_deep_read", "小为正在读取语雀文档正文和图文视频...")
            try:
                deep_read = await _read_deep_by_focus_or_query(
                    reader=self._yuque_deep_reader,
                    focus=catalog_focus,
                    question=question,
                )
            except Exception:
                deep_read = FriendV5YuqueDeepReadResult(debug={"mode": "deep_read_error"})

        if (
            not deep_read.used
            and trigger_type == "manual"
            and self._yuque_deep_reader
            and catalog_focus
            and self._should_lookup_yuque(question=question, trigger_type=trigger_type)
        ):
            yield _stage("yuque_deep_read", "小为正在读取语雀文档正文和图文视频...")
            try:
                deep_read = await _read_deep_by_focus_or_query(
                    reader=self._yuque_deep_reader,
                    focus=catalog_focus,
                    question=question,
                )
            except Exception:
                deep_read = FriendV5YuqueDeepReadResult(debug={"mode": "deep_read_error"})

        if not deep_read.used and self._should_deep_read_yuque(question=question, trigger_type=trigger_type):
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

        yield _stage("searching", "小为正在结合联网搜索梳理资料...")
        system_prompt = build_friend_v5_system_prompt()
        user_prompt = self._build_user_prompt(
            question=question,
            scene=scene,
            trigger_type=trigger_type,
            history=history,
            profile=profile,
            yuque_sources=yuque_sources,
            deep_read=deep_read,
        )
        parser = FriendV5TagStreamFilter(scene=scene)
        answer_parts: List[str] = []
        web_sources: List[FriendV5SourceItem] = []
        search_keywords: List[str] = []

        async for item in self._generator.stream(system_prompt=system_prompt, user_prompt=user_prompt):
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

        source_items = _dedupe_sources(_source_urls_to_items(parsed.source_urls), existing=web_sources)
        if source_items:
            logger.info("V5 从 [SOURCES] 块解析到 %d 个来源链接", len(source_items))
        if web_sources:
            logger.info("V5 从联网搜索响应解析到 %d 个来源链接", len(web_sources))
        merged_sources = _dedupe_sources([*web_sources, *source_items, *yuque_sources], existing=[])
        merged_keywords = search_keywords or _derive_search_keywords(question)

        catalog_tags = _catalog_matched_tags(
            question=question,
            scene=scene,
            parsed_tags=parsed.tags,
            toc_nodes=self._toc_nodes,
            focus=catalog_focus,
        )
        rhythm_tags = _apply_recommendation_tag_rhythm(
            content_tags=catalog_tags.tags,
            question=question,
            scene=scene,
            trigger_type=trigger_type,
            history=history,
        )

        payload = ChatV5DonePayload(
            answer=answer,
            tags=rhythm_tags.tags,
            sources=merged_sources,
            search_keywords=merged_keywords,
            media=deep_read.media,
            profile_fields=_profile_fields(profile),
            fallback_used=False,
            debug={
                "pipeline": "friend_v5",
                "scene": scene,
                "trigger_type": trigger_type,
                "scene_query_rewrite": scene_query_rewrite,
                "doc_deep_read_used": bool(deep_read.used),
                "doc_deep_read": deep_read.debug,
                "case_branch_used": case_branch_used,
                "conversion_state": rhythm_tags.conversion_state,
                "search_keyword_count": len(merged_keywords),
                "web_source_count": len([item for item in merged_sources if item.source_type == "web"]),
                "yuque_source_count": len([item for item in merged_sources if item.source_type == "yuque"]),
                "catalog_tag_source": "yuque_toc",
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
        yuque_instruction = (
            "已提供语雀文档正文摘录，请优先基于该摘录回答；如果摘录没有提到，不要编造。"
            if deep_read.used
            else "请联网搜索后回答。你可以把语雀链接作为补充阅读入口，但不要声称已经读过链接里的全文。"
        )
        return (
            f"【当前场景】\n{scene}\n{scene_description(scene)}\n\n"
            f"【触发方式】\n{trigger_type}\n\n"
            f"【已了解的信息】\n{_profile_block(profile)}\n\n"
            f"【最近对话】\n{chr(10).join(hist_lines) if hist_lines else '（暂无）'}\n\n"
            f"【语雀补充阅读链接】\n{chr(10).join(yuque_lines) if yuque_lines else '（本轮不提供语雀链接）'}\n\n"
            f"{deep_read.prompt_block + chr(10) + chr(10) if deep_read.used else ''}"
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
    if interests:
        lines.append(f"兴趣参考：{interests}")
    return "\n".join(lines)


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
) -> _TagRhythmResult:
    turn_index = _v5_turn_index(history)
    clean_content_tags = _clean_content_tags(content_tags)
    conversion_intents = _conversion_intents(question)
    interest_state = _interest_state(question=question, scene=scene, history=history)

    if turn_index <= 1:
        tags = clean_content_tags[:3]
        stage = "content_only"
    elif turn_index == 2:
        tags = _dedupe_tag_list([*clean_content_tags[:2], _PRODUCT_CASE_TAG])[:3]
        stage = "case_intro"
    else:
        conversion_labels = [_conversion_label(intent) for intent in conversion_intents[:2]]
        if len(conversion_labels) >= 2:
            tags = _dedupe_tag_list([*clean_content_tags[:1], *conversion_labels])[:3]
        elif conversion_labels:
            tags = _dedupe_tag_list([*clean_content_tags[:2], conversion_labels[0]])[:3]
        else:
            tags = clean_content_tags[:3]
        stage = "conversion_gradual" if conversion_labels else "content_deepen"

    # 保证每轮都有标签：不足 3 个时用场景兜底标签补齐
    if len(tags) < 3:
        existing_norms = {_norm_for_match(tag) for tag in tags}
        for fallback in fallback_tags_for_scene(scene):
            if len(tags) >= 3:
                break
            norm = _norm_for_match(fallback)
            if not norm or norm in existing_norms:
                continue
            existing_norms.add(norm)
            tags.append(fallback)

    return _TagRhythmResult(
        tags=tags,
        conversion_state={
            "turn_index": turn_index,
            "stage": stage,
            "trigger_type": trigger_type,
            "interests": interest_state,
            "conversion_intents": conversion_intents[:2],
        },
    )


def _v5_turn_index(history: Sequence[ChatMessageRow]) -> int:
    user_turns = sum(1 for row in history if getattr(row, "role", "") == "user")
    return max(1, user_turns + 1)


def _clean_content_tags(tags: Sequence[str]) -> List[str]:
    blocked_norms = {_norm_for_match(_PRODUCT_CASE_TAG), *{_norm_for_match(tag) for tag in _CONVERSION_TAGS}}
    out: List[str] = []
    seen: set[str] = set()
    for raw in tags:
        tag = str(raw or "").strip()
        norm = _norm_for_match(tag)
        if not tag or not norm or norm in blocked_norms or norm in seen:
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


def _conversion_intents(question: str) -> List[str]:
    q = _norm_for_match(question)
    intents: List[str] = []
    if re.search(r"(价格|报价|费用|收费|多少钱|预算|采购|购买|买|成本)", q):
        intents.append("price")
    if re.search(r"(内测|测试|试用|账号|申请|开通|体验|试试看|试一下)", q):
        intents.append("trial")
    return intents


def _conversion_label(intent: str) -> str:
    if intent == "trial":
        return _TRIAL_TAG
    return _PRICE_TAG


def _interest_state(*, question: str, scene: str, history: Sequence[ChatMessageRow]) -> dict[str, Any]:
    text = " ".join(
        [
            scene,
            *[
                str(getattr(row, "content", "") or "")
                for row in history[-8:]
                if getattr(row, "role", "") == "user"
            ],
            question,
        ]
    )
    topics: List[str] = []
    for label, pattern in (
        ("人工智能通识教育", r"人工智能|通识|乐高|苹果|索尼|腾讯|课程"),
        ("跨学科项目式学习", r"跨学科|项目式|项目化|pbl|STEAM"),
        ("智能招生", r"招生|获客|家长|咨询"),
        ("学校AI场景定制", r"学校|校本|实验室|定制|场景"),
        ("价格关注", r"价格|报价|费用|多少钱|预算|采购|购买"),
        ("试用关注", r"内测|测试|试用|账号|申请|体验"),
    ):
        if re.search(pattern, text, flags=re.IGNORECASE):
            topics.append(label)
    return {
        "topics": topics[:5],
        "last_user_question": (question or "").strip()[:120],
    }


def _is_product_case_request(*, question: str, trigger_type: str) -> bool:
    if trigger_type != "tag":
        return False
    return _norm_for_match(question) in _PRODUCT_CASE_TAG_NORMS


def _case_query_context(*, question: str, scene: str, history: Sequence[ChatMessageRow]) -> str:
    parts: List[str] = [scene]
    for row in list(history)[-8:]:
        content = (getattr(row, "content", "") or "").strip()
        if not content or _norm_for_match(content) in _PRODUCT_CASE_TAG_NORMS:
            continue
        if getattr(row, "role", "") == "user":
            parts.append(content)
    parts.append(question)
    return " ".join(parts).strip()


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
) -> FriendV5YuqueDeepReadResult:
    if focus and hasattr(reader, "read_toc_node"):
        result = await reader.read_toc_node(node=focus, question=question)
        if getattr(result, "used", False):
            return result
    return await reader.read(question=question)


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
    """把兜底标签映射到对应的语雀目录节点；目录为分组(无 doc_id)时下钻到首个可读子文档。"""
    title = TAG_TO_TOC_TITLE.get((question or "").strip())
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
