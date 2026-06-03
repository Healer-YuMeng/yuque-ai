from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, Tuple

from app.conversation.chat_display import display_name_for_chat
from app.conversation.interest_guide_selector import InterestGuideSelector
from app.conversation.profile_extractor import ProfileExtractor
from app.core.config import settings
from app.core.logger import get_logger
from app.data.mcp_client import MCPSearchResult, YuqueMCPClient
from app.db.profile_repository import ChatSessionProfile, ChatSessionProfileRepository
from app.db.repositories import ChatMessageRow
from app.rag.generator import Generator
from app.schemas.chat import ChatMediaBundle, ChatV2Response, GuideDocTitleNode, SourceItem
from app.service.media_answer_orchestrator import (
    _DocContext,
    collect_media_from_doc_contexts,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class V3Context:
    session_id: str
    profile: Optional[ChatSessionProfile]
    history: Sequence[ChatMessageRow]


class SalesDialogOrchestratorV3:
    def __init__(
        self,
        *,
        mcp_client: YuqueMCPClient,
        generator: Generator,
        profile_repo: ChatSessionProfileRepository,
        toc_nodes: Sequence[Dict[str, Any]],
    ) -> None:
        self._mcp_client = mcp_client
        self._generator = generator
        self._profile_repo = profile_repo
        self._extractor = ProfileExtractor()
        self._selector = InterestGuideSelector()
        self._toc_tree = _build_toc_tree(toc_nodes)

    async def answer_stream(
        self,
        *,
        question: str,
        session_id: str,
        history: Sequence[ChatMessageRow],
    ) -> AsyncIterator[dict[str, Any]]:
        sid = (session_id or "").strip()
        profile = await self._profile_repo.get_profile(session_id=sid) if sid else None
        update = await self._extractor.extract_update(question=question, history=history, current_profile=profile)
        if sid:
            # 合并兴趣：以 update 为准（若为空则不覆盖）
            merged_interests = None
            if update.interests is not None:
                base = dict(profile.interests) if profile else {}
                base.update(update.interests)
                merged_interests = base
            await self._profile_repo.upsert_profile(
                session_id=sid,
                display_name=update.display_name if update.display_name is not None else None,
                visitor_type=str(update.visitor_type) if update.visitor_type is not None else None,
                org_name=update.org_name if update.org_name is not None else None,
                interests=merged_interests,
            )
            profile = await self._profile_repo.get_profile(session_id=sid)

        vt = (profile.visitor_type if profile and profile.visitor_type else "") or ""
        interests = profile.interests if profile else {}

        # 0) 用户已点名某个主题/目录：拉正文+配图，直接讲解（不再重复给 3 选 1）
        selected_title = _match_selected_title(question=question, toc_tree=self._toc_tree)
        if selected_title:
            async for event in self._answer_with_docs(
                question=question,
                profile=profile,
                session_id=sid,
                title=selected_title,
                mode="v3_selected_topic",
                history=history,
            ):
                yield event
            return

        # 1) follow-up：基于已聚焦文档继续答（同样带配图）
        is_follow_up = _looks_like_follow_up(question, history=history)
        focused = list(profile.focused_doc_ids) if profile else []

        if is_follow_up and focused:
            yield {"event": "stage", "data": {"stage": "retrieving", "detail": "收到，我接着刚才的内容帮你补充…", "mode": "v3"}}
            doc_ctx, sources = await self._fetch_doc_contexts_by_ids(focused[:3])
            media = collect_media_from_doc_contexts(
                doc_ctx,
                question=question,
                max_images=settings.chat_v15_max_images,
                max_videos=settings.chat_v15_max_videos,
            )
            prompt = _build_sales_prompt(question=question, profile=profile, mode="follow_up", has_media=bool(media.images or media.videos))
            contexts = [_build_context_from_doc_ctx(d) for d in doc_ctx if d.body or d.snippet]
            yield {"event": "stage", "data": {"stage": "generating", "detail": "我整理下要点…", "mode": "v3"}}
            async for token in self._generator.stream_generate(
                question=prompt,
                contexts=contexts or ["（当前未命中足够上下文，请先给出下一步澄清问题。）"],
                sources=sources,
                visitor_sales=True,
            ):
                yield {"event": "token", "data": {"token": token}}
            related = self._related_topic_picks(current_title=question, exclude=set(), limit=2)
            yield {
                "event": "done",
                "data": ChatV2Response(
                    answer="",
                    sources=sources,
                    fallback_used=not bool(sources),
                    debug={"mode": "v3_follow_up", "related_topics": [p.title for p in related]},
                    media=media,
                    lead_nudge_triggered=False,
                ).model_dump(),
            }
            return

        # 2) browse/不明确：兴趣驱动推荐 3 个 TOC 标题（话术自然、不重复上一轮）
        yield {"event": "stage", "data": {"stage": "guiding", "detail": "正在为你挑选合适的内容方向…", "mode": "v3"}}
        exclude = _recent_suggested_titles(history)
        picks = self._selector.pick_top3(
            question=question,
            toc_nodes=self._toc_tree,
            interests=interests,
            visitor_type=vt,
            exclude_titles=list(exclude),
        )
        if picks:
            msg = _build_guide_message(question=question, profile=profile, picks=picks, history=history)
            for ch in msg:
                yield {"event": "token", "data": {"token": ch}}
            yield {"event": "done", "data": ChatV2Response(answer=msg, sources=[], fallback_used=True, debug={"mode": "v3_guide"}, media=ChatMediaBundle(), lead_nudge_triggered=False).model_dump()}
            return

        # 3) 兜底：如果 toc 没有可用 pick，则回退成简短澄清
        fallback = _build_clarify_message(profile=profile)
        for ch in fallback:
            yield {"event": "token", "data": {"token": ch}}
        yield {"event": "done", "data": ChatV2Response(answer=fallback, sources=[], fallback_used=True, debug={"mode": "v3_clarify"}, media=ChatMediaBundle(), lead_nudge_triggered=False).model_dump()}

    async def _answer_with_docs(
        self,
        *,
        question: str,
        profile: Optional[ChatSessionProfile],
        session_id: str,
        title: str,
        mode: str,
        history: Sequence[ChatMessageRow],
    ) -> AsyncIterator[dict[str, Any]]:
        yield {"event": "stage", "data": {"stage": "retrieving", "detail": "收到，我先把相关内容整理好…", "mode": "v3"}}
        doc_ctx, sources = await self._retrieve_docs_for_title(title)
        media = collect_media_from_doc_contexts(
            doc_ctx,
            question=question,
            max_images=settings.chat_v15_max_images,
            max_videos=settings.chat_v15_max_videos,
        )
        sid = (session_id or "").strip()
        if sid and doc_ctx:
            await self._profile_repo.touch_focus_docs(session_id=sid, doc_ids=[d.doc_id for d in doc_ctx if d.doc_id])
            profile = await self._profile_repo.get_profile(session_id=sid)
        prompt = _build_sales_prompt(
            question=question,
            profile=profile,
            mode="selected_topic",
            has_media=bool(media.images or media.videos),
        )
        contexts = [_build_context_from_doc_ctx(d) for d in doc_ctx if d.body or d.snippet]
        yield {"event": "stage", "data": {"stage": "generating", "detail": "我用图文帮你快速讲清楚…", "mode": "v3"}}
        async for token in self._generator.stream_generate(
            question=prompt,
            contexts=contexts or ["（当前未检索到可用摘录，请用一句话说明你最关心的场景。）"],
            sources=sources,
            visitor_sales=True,
        ):
            yield {"event": "token", "data": {"token": token}}
        related = self._related_topic_picks(current_title=title, exclude={title}, limit=2)
        yield {
            "event": "done",
            "data": ChatV2Response(
                answer="",
                sources=sources,
                fallback_used=not bool(sources),
                debug={
                    "mode": mode,
                    "selected_title": title,
                    "related_topics": [p.title for p in related],
                },
                media=media,
                lead_nudge_triggered=False,
            ).model_dump(),
        }

    def _related_topic_picks(self, *, current_title: str, exclude: set[str], limit: int = 2) -> List[Any]:
        picks = self._selector.pick_top3(
            question=current_title or "相关内容",
            toc_nodes=self._toc_tree,
            interests=None,
            visitor_type=None,
            exclude_titles=list(exclude),
        )
        return picks[: max(0, int(limit))]

    async def _fetch_docs(self, doc_ids: Sequence[str]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for did in doc_ids:
            body = ""
            try:
                body = await self._mcp_client.get_doc(str(did))
            except Exception as exc:
                logger.warning("v3_get_doc_failed doc_id=%s err=%s", did, exc)
            title = f"语雀文档 {did}"
            url = ""
            out.append(
                {
                    "context": f"文档ID：{did}\n{(body or '').strip()[:2800]}",
                    "source": {"title": title, "url": url or None, "source_type": "mcp", "snippet": (body or '')[:200], "doc_id": str(did)},
                }
            )
        return out

    async def _retrieve_docs_for_title(self, title: str) -> Tuple[List[_DocContext], List[SourceItem]]:
        if not self._mcp_client.enabled:
            return [], []
        query = (title or "").strip()
        if not query:
            return [], []

        hits: List[MCPSearchResult] = []
        node = _find_toc_node_by_title(self._toc_tree, query)
        if node and node.doc_id is not None:
            did = str(node.doc_id)
            hits.append(
                MCPSearchResult(
                    doc_id=did,
                    title=node.title or query,
                    url=(node.url or ""),
                    snippet="",
                )
            )
        try:
            searched = await self._mcp_client.search(query)
        except Exception as exc:
            logger.warning("v3_mcp_search_failed query=%r err=%s", query, exc)
            searched = []
        seen_ids: set[str] = set()
        for h in searched:
            did = (h.doc_id or "").strip()
            if not did or did in seen_ids:
                continue
            seen_ids.add(did)
            hits.append(h)
            if len(hits) >= settings.chat_v15_max_docs:
                break

        async def _fetch_one(item: MCPSearchResult) -> _DocContext:
            body = ""
            try:
                body = await self._mcp_client.get_doc(item.doc_id)
            except Exception as exc:
                logger.warning("v3_get_doc_failed doc_id=%s err=%s", item.doc_id, exc)
            url = (item.url or "").strip()
            return _DocContext(
                doc_id=item.doc_id,
                title=item.title or query,
                url=url,
                snippet=(item.snippet or "")[:400],
                body=(body or "").strip(),
            )

        doc_ctx = await asyncio.gather(*[_fetch_one(h) for h in hits[: settings.chat_v15_max_docs]], return_exceptions=False)
        doc_ctx = [d for d in doc_ctx if d.body or d.snippet]
        sources: List[SourceItem] = []
        for d in doc_ctx:
            sources.append(
                SourceItem(
                    title=d.title or query,
                    url=(d.url or None),
                    source_type="mcp",
                    snippet=(d.snippet or d.body or "")[:200] or None,
                    doc_id=d.doc_id or None,
                )
            )
        return doc_ctx, sources

    async def _fetch_doc_contexts_by_ids(self, doc_ids: Sequence[str]) -> Tuple[List[_DocContext], List[SourceItem]]:
        out: List[_DocContext] = []
        sources: List[SourceItem] = []
        for did in doc_ids:
            body = ""
            try:
                body = await self._mcp_client.get_doc(str(did))
            except Exception as exc:
                logger.warning("v3_get_doc_failed doc_id=%s err=%s", did, exc)
            ctx = _DocContext(doc_id=str(did), title=f"文档 {did}", url="", snippet="", body=(body or "").strip())
            if ctx.body:
                out.append(ctx)
                sources.append(
                    SourceItem(
                        title=ctx.title,
                        url=None,
                        source_type="mcp",
                        snippet=ctx.body[:200],
                        doc_id=str(did),
                    )
                )
        return out, sources


def _looks_like_follow_up(question: str, *, history: Sequence[ChatMessageRow]) -> bool:
    q = (question or "").strip()
    if not q:
        return False
    if any(k in q for k in ("刚才", "上面", "这个", "那你说的", "继续", "再展开", "详细讲讲")):
        return True
    # 上一轮助手说过某个明显的产品名/模块名，用户又追问“怎么/如何/步骤”
    if any(k in q for k in ("怎么", "如何", "步骤", "流程")):
        for row in reversed(history[-6:]):
            if row.role != "assistant":
                continue
            t = (row.content or "").strip()
            if not t:
                continue
            if "IDEAS-PBL" in t or "使用指南" in t or "平台介绍" in t:
                return True
            break
    return False


def _build_guide_message(
    *,
    question: str,
    profile: Optional[ChatSessionProfile],
    picks: Sequence[Any],
    history: Sequence[ChatMessageRow],
) -> str:
    _ = question
    name = _display_name_for_chat(profile)
    vt = (profile.visitor_type if profile else "") or ""
    greet = _greet_line(name=name, visitor_type=vt)
    recently_guided = _recently_guided(history)
    intro_only = _is_intro_only(question)
    if intro_only and name:
        preface = f"{greet}很高兴认识您。结合您这边的情况，我先帮您挑 3 个同事最常问的方向："
    elif not recently_guided:
        preface = f"{greet}我先帮您挑 3 个最可能用得上的方向："
    else:
        preface = f"{greet}那我们换个角度，您也可以看看下面这几个："
    lines = [preface]
    for i, p in enumerate(picks[:3], start=1):
        title = getattr(p, "title", "") or ""
        lines.append(f"{i}. {title}")
    lines.append("")
    lines.append("您更想先聊哪一个？直接说名称就行。")
    return "\n".join(lines).strip()


def _build_clarify_message(*, profile: Optional[ChatSessionProfile]) -> str:
    vt = (profile.visitor_type if profile else "") or ""
    if vt:
        return "我可以帮你快速定位到合适的内容方向。你更想先了解：怎么用、有哪些案例，还是试用/部署相关？"
    return "我可以帮你快速定位到合适的内容方向。你方便说下你是校长/老师/家长/学生吗？以及你更想先了解怎么用、案例还是试用/部署？"


def _build_sales_prompt(
    *,
    question: str,
    profile: Optional[ChatSessionProfile],
    mode: str,
    has_media: bool = False,
) -> str:
    name = _display_name_for_chat(profile)
    vt = (profile.visitor_type if profile else "") or ""
    org = (profile.org_name if profile else "") or ""
    persona = (
        "你是「有为人工智能教育平台」的在线销售咨询顾问，名字叫「小为顾问」。"
        "请用自然、亲切、专业的中文沟通，像真人顾问，而不是在朗读资料。"
    )
    memory = []
    if name:
        memory.append(f"称呼：{name}")
    if vt:
        memory.append(f"身份（仅内部参考，勿在回复中复述）：{vt}")
    if org:
        memory.append(f"单位（仅内部参考，勿在回复中写出学校/机构名）：{org}")
    mem_block = ("\n".join(memory) + "\n") if memory else ""
    media_hint = (
        "正文尽量精炼（约90-150字）；若有配图，先用1句引导语承接，再结合图片简短说明，不要只丢素材。"
        if has_media
        else "正文尽量精炼（约90-150字）。"
    )
    return (
        f"{persona}\n"
        "请优先回答用户问题本身，像真人销售讲解产品，避免模板化开场。\n"
        "禁止在回复中出现：语雀、知识库、资料库、目录结构、文档标题列表等字样；"
        "不要复述用户的单位/学校名称；若已知称呼，可在开头自然称呼一次。\n"
        f"{media_hint}\n"
        "输出结构固定为：先1-2句承接用户，再1句总领，再用3-4条 `- **关键词**：说明` 讲重点，最后1句自然追问。\n"
        "如果上下文不足，只问1个最关键的问题。\n"
        f"{mem_block}\n"
        f"用户问题：{question}\n"
        f"模式：{mode}"
    )


def _match_selected_title(*, question: str, toc_tree: Sequence[GuideDocTitleNode]) -> str:
    q = (question or "").strip()
    if not q:
        return ""
    core = _strip_topic_prefix(q)
    probe = core or q
    # 明确想了解某主题，或问句与某 TOC 标题高度相似
    has_intent = bool(core) or any(k in q for k in ("了解", "介绍", "讲讲", "看看", "想看", "课程", "通识", "平台"))
    if not has_intent:
        return ""
    best: Tuple[int, str] | None = None

    def _walk(nodes: Sequence[GuideDocTitleNode]) -> None:
        nonlocal best
        for n in nodes:
            t = (n.title or "").strip()
            if not t:
                continue
            score = _title_match_score(probe, t)
            if score > 0 and (best is None or score > best[0]):
                best = (score, t)
            if n.children:
                _walk(n.children)

    _walk(toc_tree)
    if not best:
        return ""
    return best[1] if best[0] >= 55 else ""


def _build_context_from_doc_ctx(doc: _DocContext) -> str:
    body = (doc.body or "").strip()
    snippet = (doc.snippet or "").strip()
    chunk = body[:3200] if body else snippet[:1200]
    return f"文档标题：{doc.title}\n正文摘录：\n{chunk}"


def _strip_topic_prefix(q: str) -> str:
    s = (q or "").strip()
    prefixes = (
        "我想看看",
        "我想了解下",
        "我想了解",
        "我想看",
        "帮我看看",
        "帮我讲讲",
        "介绍一下",
        "介绍",
        "讲讲",
        "看看",
    )
    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if s.startswith(p):
                s = s[len(p) :].strip("《》 \t，,。！？?、")
                changed = True
    return s.strip("《》 ")


def _norm_title_key(s: str) -> str:
    t = (s or "").strip().lower()
    for suffix in ("介绍", "课程", "方案", "指南"):
        if t.endswith(suffix) and len(t) > len(suffix) + 1:
            t = t[: -len(suffix)]
    return t


def _title_match_score(query: str, title: str) -> int:
    q = _norm_title_key(query)
    t = _norm_title_key(title)
    if not q or not t:
        return 0
    score = 0
    if q == t:
        return 300 + len(t)
    if q in t or t in q:
        score += 180 + min(len(q), len(t))
    # 片段重叠：通识课 ↔ 人工智能通识课程
    for n in (4, 3, 2):
        if len(q) >= n:
            for i in range(0, len(q) - n + 1):
                frag = q[i : i + n]
                if frag and frag in t:
                    score += 12 + n
    for w in re.split(r"[\s,，。！？?、]+", query):
        w2 = (w or "").strip()
        if len(w2) >= 2 and w2 in title:
            score += 10
    return score


def _display_name_for_chat(profile: Optional[ChatSessionProfile]) -> str:
    return display_name_for_chat(profile)


def _greet_line(*, name: str, visitor_type: str) -> str:
    if name:
        return f"{name}，您好！"
    vt = (visitor_type or "").strip()
    if vt == "teacher":
        return "老师您好！"
    if vt == "parent":
        return "家长您好！"
    if vt == "student":
        return "同学你好！"
    return "您好！"


def _is_intro_only(question: str) -> bool:
    q = (question or "").strip()
    if not q:
        return False
    has_identity = any(k in q for k in ("我是", "我时", "我叫", "来自", "老师", "校长", "家长", "同学"))
    has_topic = any(
        k in q
        for k in ("了解", "看看", "介绍", "平台", "课程", "通识", "案例", "PBL", "部署", "试用", "价格")
    )
    return has_identity and not has_topic


def _recently_guided(history: Sequence[ChatMessageRow]) -> bool:
    for row in reversed(history[-4:]):
        if row.role != "assistant":
            continue
        t = (row.content or "").strip()
        return any(x in t for x in ("更想先聊哪一个", "最常问的方向", "最可能用得上的方向"))
    return False


def _recent_suggested_titles(history: Sequence[ChatMessageRow]) -> set[str]:
    out: set[str] = set()
    for row in reversed(history[-6:]):
        if row.role != "assistant":
            continue
        for line in (row.content or "").splitlines():
            m = re.match(r"^\s*\d+[\.\)、]\s*(.+?)\s*$", line.strip())
            if m:
                title = m.group(1).strip().strip("《》")
                if title:
                    out.add(title)
        if out:
            break
    return out


def _find_toc_node_by_title(
    toc_tree: Sequence[GuideDocTitleNode],
    title: str,
) -> Optional[GuideDocTitleNode]:
    target = (title or "").strip()
    if not target:
        return None
    best: Tuple[int, GuideDocTitleNode] | None = None

    def _walk(nodes: Sequence[GuideDocTitleNode]) -> None:
        nonlocal best
        for n in nodes:
            t = (n.title or "").strip()
            if not t:
                continue
            sc = _title_match_score(target, t)
            if sc > 0 and (best is None or sc > best[0]):
                best = (sc, n)
            if n.children:
                _walk(n.children)

    _walk(toc_tree)
    if not best or best[0] < 55:
        return None
    return best[1]


def _build_toc_tree(raw_nodes: Sequence[Dict[str, Any]]) -> List[GuideDocTitleNode]:
    # raw_nodes 来自 QAService 预热的 toc_nodes_data（扁平）
    items: List[GuideDocTitleNode] = []
    by_uuid: Dict[str, GuideDocTitleNode] = {}
    for x in raw_nodes or []:
        title = str(x.get("title") or "").strip()
        if not title:
            continue
        node = GuideDocTitleNode(
            uuid=str(x.get("uuid") or ""),
            title=title,
            level=int(x.get("level") or 1),
            parent_uuid=str(x.get("parent_uuid") or ""),
            node_type=str(x.get("node_type") or ""),
            url=(str(x.get("url") or "").strip() or None),
            doc_id=(int(x.get("doc_id")) if x.get("doc_id") is not None else None),
            children=[],
        )
        items.append(node)
        if node.uuid and node.uuid not in by_uuid:
            by_uuid[node.uuid] = node
    roots: List[GuideDocTitleNode] = []
    for node in items:
        pu = node.parent_uuid
        if pu and pu in by_uuid and pu != node.uuid:
            by_uuid[pu].children.append(node)
        else:
            roots.append(node)
    return roots
