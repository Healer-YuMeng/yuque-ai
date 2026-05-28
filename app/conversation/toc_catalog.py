from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.schemas.chat import GuideDocTitleNode


@dataclass
class CatalogNode:
    uuid: str
    title: str
    level: int
    parent_uuid: str
    node_type: str
    url: Optional[str]
    doc_id: Optional[int]
    path_titles: List[str] = field(default_factory=list)
    children: List["CatalogNode"] = field(default_factory=list)


_RESET_PHRASES = ("回到首页", "重新开始", "从头开始", "返回首页", "重置对话")


class TocCatalogIndex:
    """语雀 TOC 索引：路径、子节点、同级关联、范围内匹配。"""

    def __init__(self, raw_nodes: Sequence[Dict[str, Any]]) -> None:
        self._roots: List[CatalogNode] = _build_catalog_tree(raw_nodes)
        self._by_uuid: Dict[str, CatalogNode] = {}
        self._walk_index(self._roots)

    @property
    def roots(self) -> List[CatalogNode]:
        return list(self._roots)

    def get(self, uuid: str) -> Optional[CatalogNode]:
        return self._by_uuid.get((uuid or "").strip())

    def dialog_level(self, node: Optional[CatalogNode]) -> int:
        """0=根 1=一级模块 2=二级 3=深度内容（路径≥3）。"""
        if not node:
            return 0
        depth = len(node.path_titles)
        if depth <= 0:
            return 0
        if depth == 1:
            return 1
        if depth == 2:
            return 2
        return 3

    def is_reset_intent(self, question: str) -> bool:
        q = (question or "").strip()
        return any(p in q for p in _RESET_PHRASES)

    def match_node(
        self,
        question: str,
        *,
        current: Optional[CatalogNode],
        prefer_subtree: bool = True,
    ) -> Optional[CatalogNode]:
        q = _strip_intent_prefix((question or "").strip())
        if not q:
            return None
        best: Tuple[int, CatalogNode] | None = None
        for node in self._by_uuid.values():
            score = _title_match_score(q, node.title)
            if score <= 0:
                continue
            if prefer_subtree and current:
                if _is_descendant_or_self(node, current):
                    score += 80
                elif current.parent_uuid and node.parent_uuid == current.parent_uuid:
                    score += 40
            if best is None or score > best[0]:
                best = (score, node)
        if not best or best[0] < 55:
            return None
        return best[1]

    def can_advance(self, current: Optional[CatalogNode], target: CatalogNode) -> bool:
        if not current:
            return True
        cur_depth = len(current.path_titles)
        new_depth = len(target.path_titles)
        if new_depth >= cur_depth:
            return True
        return False

    def children_of(self, node: Optional[CatalogNode]) -> List[CatalogNode]:
        if not node:
            return list(self._roots)
        return list(node.children)

    def related_in_catalog(self, node: CatalogNode, *, limit: int = 3) -> List[CatalogNode]:
        """同级 / 父级其它子节点；不跳到根目录其它一级模块（除非同级）。"""
        out: List[CatalogNode] = []
        parent = self.get(node.parent_uuid) if node.parent_uuid else None
        if parent:
            for sib in parent.children:
                if sib.uuid != node.uuid:
                    out.append(sib)
        for child in node.children:
            if child.title not in {x.title for x in out}:
                out.append(child)
        return out[: max(1, int(limit))]

    def _walk_index(self, nodes: List[CatalogNode]) -> None:
        for n in nodes:
            if n.uuid:
                self._by_uuid[n.uuid] = n
            if n.children:
                self._walk_index(n.children)


def _build_catalog_tree(raw_nodes: Sequence[Dict[str, Any]]) -> List[CatalogNode]:
    items: List[CatalogNode] = []
    by_uuid: Dict[str, CatalogNode] = {}
    for x in raw_nodes or []:
        title = str(x.get("title") or "").strip()
        if not title:
            continue
        node = CatalogNode(
            uuid=str(x.get("uuid") or ""),
            title=title,
            level=int(x.get("level") or 1),
            parent_uuid=str(x.get("parent_uuid") or ""),
            node_type=str(x.get("node_type") or ""),
            url=(str(x.get("url") or "").strip() or None),
            doc_id=(int(x.get("doc_id")) if x.get("doc_id") is not None else None),
            path_titles=[],
            children=[],
        )
        items.append(node)
        if node.uuid and node.uuid not in by_uuid:
            by_uuid[node.uuid] = node
    roots: List[CatalogNode] = []
    for node in items:
        pu = node.parent_uuid
        if pu and pu in by_uuid and pu != node.uuid:
            by_uuid[pu].children.append(node)
        else:
            roots.append(node)

    def _fill_path(ns: List[CatalogNode], prefix: List[str]) -> None:
        for n in ns:
            n.path_titles = prefix + [n.title]
            if n.children:
                _fill_path(n.children, n.path_titles)

    _fill_path(roots, [])
    return roots


def _is_descendant_or_self(node: CatalogNode, ancestor: CatalogNode) -> bool:
    if node.uuid == ancestor.uuid:
        return True
    path = " / ".join(node.path_titles)
    ap = " / ".join(ancestor.path_titles)
    return path.startswith(ap + " /") or path == ap


def _strip_intent_prefix(q: str) -> str:
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
    if q == t:
        return 300 + len(t)
    if q in t or t in q:
        return 180 + min(len(q), len(t))
    score = 0
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


def catalog_node_to_guide(node: CatalogNode) -> GuideDocTitleNode:
    return GuideDocTitleNode(
        uuid=node.uuid,
        title=node.title,
        level=node.level,
        parent_uuid=node.parent_uuid,
        node_type=node.node_type,
        url=node.url,
        doc_id=node.doc_id,
        children=[catalog_node_to_guide(c) for c in node.children],
    )
