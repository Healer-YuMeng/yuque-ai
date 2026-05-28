from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from app.schemas.chat import GuideDocTitleNode


@dataclass(frozen=True)
class GuidePick:
    title: str
    reason: str = ""


class InterestGuideSelector:
    """从 TOC/标题树中按“兴趣点 + 角色偏好 + 当前问句”选出 3 个最相关方向。"""

    def pick_top3(
        self,
        *,
        question: str,
        toc_nodes: Sequence[GuideDocTitleNode],
        interests: Dict[str, Any] | None,
        visitor_type: str | None,
        exclude_titles: Sequence[str] | None = None,
    ) -> List[GuidePick]:
        q = (question or "").strip().lower()
        flat = _flatten_titles(toc_nodes)
        if not flat:
            return []
        keys = _interest_keys(interests)
        excluded = {str(x or "").strip() for x in (exclude_titles or []) if str(x or "").strip()}
        scored: List[Tuple[int, GuidePick]] = []
        for title, path in flat:
            if title in excluded:
                continue
            t = title.lower()
            score = 0
            # 当前问句匹配
            if t and t in q:
                score += 120
            if len(q) >= 2 and q in t:
                score += 30
            # 兴趣点匹配
            for k in keys:
                if k and k in t:
                    score += 22
            # 角色偏好加权（极简）
            score += _role_bias_score(title=title, visitor_type=visitor_type)
            # 路径层级轻微偏好：更像“模块名”的上层标题更适合引导
            score += 8 if len(path) <= 2 else 2
            if score <= 0:
                continue
            reason = _build_reason(title=title, keys=keys, visitor_type=visitor_type)
            scored.append((score, GuidePick(title=title, reason=reason)))

        scored.sort(key=lambda x: x[0], reverse=True)
        out: List[GuidePick] = []
        seen: set[str] = set()
        for _, pick in scored:
            key = pick.title.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(pick)
            if len(out) >= 3:
                break

        # 兜底：当问句太空泛时，按常见模块给一组稳定推荐（但不是“固定模板”）
        if len(out) < 3:
            fallback = _fallback_titles(flat)
            for t in fallback:
                if t in seen:
                    continue
                out.append(GuidePick(title=t, reason="常见咨询方向"))
                seen.add(t)
                if len(out) >= 3:
                    break
        return out[:3]


def _flatten_titles(nodes: Sequence[GuideDocTitleNode]) -> List[Tuple[str, List[str]]]:
    out: List[Tuple[str, List[str]]] = []

    def _walk(ns: Sequence[GuideDocTitleNode], path: List[str]) -> None:
        for n in ns:
            title = (n.title or "").strip()
            if not title:
                continue
            next_path = path + [title]
            out.append((title, next_path))
            if n.children:
                _walk(n.children, next_path)

    _walk(nodes, [])
    return out


def _interest_keys(interests: Dict[str, Any] | None) -> List[str]:
    if not interests or not isinstance(interests, dict):
        return []
    # 按 score 降序取前 5 个
    items: List[Tuple[int, str]] = []
    for k, v in interests.items():
        key = str(k or "").strip().lower()
        if not key or len(key) < 2:
            continue
        score = 1
        if isinstance(v, dict):
            try:
                score = int(v.get("score") or 1)
            except Exception:
                score = 1
        items.append((score, key))
    items.sort(key=lambda x: x[0], reverse=True)
    return [k for _, k in items[:5]]


def _role_bias_score(*, title: str, visitor_type: str | None) -> int:
    t = (title or "")
    vt = (visitor_type or "").strip()
    if vt == "teacher":
        if any(k in t for k in ("课堂", "教学", "备课", "作业", "评价", "教案")):
            return 18
    if vt == "parent":
        if any(k in t for k in ("家长", "孩子", "成长", "学习效果", "使用门槛")):
            return 18
    if vt == "institution_decision_maker":
        if any(k in t for k in ("部署", "方案", "采购", "落地", "合作", "投入产出")):
            return 18
    return 0


def _build_reason(*, title: str, keys: Sequence[str], visitor_type: str | None) -> str:
    # 理由仅用于内部排序/debug，不直接展示给访客（避免重复括号话术）
    _ = (title, keys, visitor_type)
    return ""


def _fallback_titles(flat: Sequence[Tuple[str, List[str]]]) -> List[str]:
    titles = [t for t, _ in flat]
    prefer_keys = ("平台介绍", "使用指南", "案例", "社区")
    out: List[str] = []
    seen: set[str] = set()
    for key in prefer_keys:
        for t in titles:
            if key in t and t not in seen:
                out.append(t)
                seen.add(t)
                break
    # 再补齐
    for t in titles:
        if t in seen:
            continue
        out.append(t)
        seen.add(t)
        if len(out) >= 3:
            break
    return out[:3]

