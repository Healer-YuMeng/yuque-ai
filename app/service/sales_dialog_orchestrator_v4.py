from __future__ import annotations

import asyncio
import re
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence

from app.conversation.catalog_state_machine import (
    CatalogDialogState,
    CatalogStateMachine,

)
from app.conversation.chat_display import display_name_for_chat
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
from app.service.v4_vision_enrichment import enrich_doc_contexts_with_vision

logger = get_logger(__name__)


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

    async def answer_stream(
        self,
        *,
        question: str,
        session_id: str,
        history: Sequence[ChatMessageRow],
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

        new_state, anchor, action = self._fsm.apply_user_turn(question=question, state=catalog_state)
        if action == "reset":
            new_state = CatalogDialogState()
            anchor = None

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
            ):
                yield event
            return

        # 状态 0/1：目录内自然引导（仅当前层级子节点，不跳回根三大项）
        if self._fsm.should_show_root_guide(new_state) or new_state.dialog_level <= 1:
            candidates = self._fsm.guide_candidates(new_state, anchor)
            if candidates:
                msg = _build_natural_guide(profile=profile, state=new_state, candidates=candidates, history=history)
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
    ) -> AsyncIterator[dict[str, Any]]:
        yield {"event": "stage", "data": {"stage": "retrieving", "detail": "正在整理相关内容…", "mode": "v4"}}
        related_nodes = self._catalog.related_in_catalog(node, limit=3)
        trace = TurnTraceBuilder(
            pipeline="v4_content",
            catalog_path=node.path_titles,
            dialog_level=state.dialog_level,
        )
        doc_ctx, sources = await self._retrieve_for_nodes(
            [node] + related_nodes,
            catalog_path=node.path_titles,
            trace=trace,
            primary_title=node.title,
        )
        primary_doc = _pick_primary_doc(doc_ctx, anchor_title=node.title)
        show_media = _should_show_media(question=question, node=node, primary=primary_doc)
        media = ChatMediaBundle()
        if show_media and primary_doc:
            media = apply_yuque_proxy_to_media(
                collect_media_from_doc_contexts(
                    [primary_doc],
                    question=question,
                    max_images=min(3, settings.chat_v15_max_images),
                    max_videos=settings.chat_v15_max_videos,
                    primary_doc_title=node.title,
                )
            )
        sid = (session_id or "").strip()
        if sid and doc_ctx:
            await self._profile_repo.touch_focus_docs(session_id=sid, doc_ids=[d.doc_id for d in doc_ctx if d.doc_id])
            profile = await self._profile_repo.get_profile(session_id=sid)

        path_label = " / ".join(node.path_titles)
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
        vision_lines: List[str] = []
        vision_dbg: Dict[str, Any] = {}
        if show_media and primary_doc:
            vision_lines, vision_dbg = await enrich_doc_contexts_with_vision([primary_doc], question=question)
        if vision_lines:
            contexts.append("\n".join(vision_lines))
            yield {
                "event": "stage",
                "data": {
                    "stage": "vision",
                    "detail": f"已识读 {vision_dbg.get('vision_images_used', 0)} 张插图…",
                    "mode": "v4",
                },
            }
        yield {"event": "stage", "data": {"stage": "generating", "detail": "正在组织讲解…", "mode": "v4"}}
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
        tail += lead_end.append_text
        if tail:
            for ch in tail:
                yield {"event": "token", "data": {"token": ch}}

        dbg: Dict[str, Any] = {
            "mode": "v4_content",
            "dialog_level": state.dialog_level,
            "path": node.path_titles,
            "related": [n.title for n in related_nodes],
            "lead_nudge_reason": lead_end.nudge.reason or "",
            "contact_detected": lead_contact_detected,
        }
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
    ) -> tuple[List[_DocContext], List[SourceItem]]:
        hits: List[MCPSearchResult] = []
        seen: set[str] = set()
        path_prefix = " / ".join(catalog_path)

        for node in nodes:
            if node.doc_id is not None:
                did = str(node.doc_id)
                if did not in seen:
                    seen.add(did)
                    hits.append(
                        MCPSearchResult(
                            doc_id=did,
                            title=node.title,
                            url=(node.url or ""),
                            snippet="",
                        )
                    )
            query = f"{path_prefix} {node.title}".strip()
            try:
                searched = await self._mcp_client.search(query)
            except Exception as exc:
                logger.warning("v4_mcp_search_failed query=%r err=%s", query, exc)
                searched = []
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

        async def _fetch(item: MCPSearchResult) -> _DocContext:
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
            return _DocContext(
                doc_id=item.doc_id,
                title=item.title or "",
                url=(item.url or "").strip(),
                snippet=(item.snippet or "")[:400],
                body=(body or "").strip(),
            )

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
        return doc_ctx, sources


def _build_tagged_context(doc: _DocContext, *, is_primary: bool = True) -> str:
    body = (doc.body or "").strip()
    snippet = (doc.snippet or "").strip()
    chunk = body[:6000] if body else snippet[:2000]
    tag = "【主文档】" if is_primary else "【关联文档·可补充细节】"
    return f"{tag}{doc.title}\n文档标题：{doc.title}\n正文摘录：\n{chunk}"


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
    mem = []
    if name:
        mem.append(f"称呼：{name}")
    if vt:
        mem.append(f"身份（内部参考，勿复述）：{vt}")
    if org:
        mem.append(f"单位（内部参考，勿写出）：{org}")
    wants_visual = any(k in (question or "") for k in ("有图", "看图", "图片", "截图", "示意图", "流程图"))
    media_hint = (
        "配图已在界面展示，正文约60-100字即可；禁止输出图片URL；禁止说「没有图片/无法展示图片」。"
        if has_media
        else ("用户想看图，若上下文含插图描述请用文字说明；若无素材再说明。" if wants_visual else "正文精炼（约80-120字）。")
    )
    related = "、".join(related_titles[:3]) if related_titles else ""
    role_hint = _role_answer_hint(vt)
    hist_block = _format_history_for_prompt(history)
    already_asked_identity = _history_has_identity_question(hist_block)
    identity_rule = (
        "" if not already_asked_identity
        else "注意：本轮对话已询问过学科/年级/身份，不要再重复追问这类信息。\n"
    )
    repeat_topic_rule = ""
    if hist_block and question.strip() in hist_block:
        repeat_topic_rule = "用户在重复同一问题，请在已讲内容基础上补充细节或案例，禁止原样复述前次回答。\n"
    skill_part = f"{skill_instructions}\n" if skill_instructions.strip() else ""
    return (
        "你是有为人工智能教育平台的销售顾问，语气自然亲切。\n"
        f"当前目录位置：{catalog_path}（对话层级 {dialog_level}）。\n"
        f"{skill_part}"
        "规则：\n"
        "- 只基于给定上下文回答；禁止编造目录外模块；\n"
        "- 禁止出现语雀/知识库/目录结构等字样；\n"
        "- 禁止说「不知道您是谁/看不到身份信息」——下方「称呼/身份」仅供你组织回答；\n"
        "- 不要输出「同模块里还可以看看：XX」这类固定模板句；\n"
        "- 每场对话中，学科/年级类信息最多询问1次；联系方式已留存则不再索要；\n"
        "- 整场对话留资/试用引导最多出现1次，且不在内容讲解中间插入；\n"
        "- 禁止推荐当前目录节点之外的其他一级模块（如「智能招生」）；\n"
        "- 无需每轮都问好，不要重复前次寒暄。\n"
        f"{identity_rule}"
        f"{repeat_topic_rule}"
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


def _should_show_media(*, question: str, node: CatalogNode, primary: Optional[_DocContext]) -> bool:
    if not primary or not doc_body_has_images(primary):
        return False
    q = (question or "").strip()
    if any(k in q for k in ("有图", "看图", "图片", "截图", "示意图", "流程图")):
        return True
    if any(k in q for k in ("测试账号", "试用账号", "申请测试", "开通试用")) and not any(
        k in q for k in ("介绍", "了解", "看看", "功能", "课程", "平台")
    ):
        return False
    return _title_match_score(q, node.title) >= 55 or _title_match_score(node.title, q) >= 55


def _history_has_identity_question(hist_block: str) -> bool:
    """检查历史中是否已经询问过学科/年级等身份信息。"""
    triggers = ("学科", "年级", "带几年级", "教什么", "您是哪个", "负责哪个", "教哪个")
    return any(k in (hist_block or "") for k in triggers)


def _user_wants_more(question: str) -> bool:
    """用户主动表达想继续了解其它内容时才返回 True，避免每轮自动追加同模块推荐。"""
    q = (question or "").strip()
    triggers = ("还有", "其他", "别的", "更多", "还能", "还想", "再推荐", "推荐", "类似", "还有没有", "看看其他")
    return any(k in q for k in triggers)


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
