from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.conversation.toc_catalog import CatalogNode, TocCatalogIndex


@dataclass
class CatalogDialogState:
    node_uuid: str = ""
    path_titles: List[str] = field(default_factory=list)
    root_guide_shown: bool = False
    dialog_level: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_uuid": self.node_uuid,
            "path_titles": list(self.path_titles),
            "root_guide_shown": bool(self.root_guide_shown),
            "dialog_level": int(self.dialog_level),
        }

    @staticmethod
    def from_dict(raw: Any) -> "CatalogDialogState":
        if not isinstance(raw, dict):
            return CatalogDialogState()
        return CatalogDialogState(
            node_uuid=str(raw.get("node_uuid") or ""),
            path_titles=[str(x) for x in (raw.get("path_titles") or []) if str(x).strip()],
            root_guide_shown=bool(raw.get("root_guide_shown")),
            dialog_level=int(raw.get("dialog_level") or 0),
        )


def parse_catalog_state_json(raw: Any) -> CatalogDialogState:
    if raw is None:
        return CatalogDialogState()
    if isinstance(raw, dict):
        return CatalogDialogState.from_dict(raw)
    try:
        s = str(raw or "").strip()
        if not s:
            return CatalogDialogState()
        return CatalogDialogState.from_dict(json.loads(s))
    except Exception:
        return CatalogDialogState()


def dump_catalog_state_json(state: CatalogDialogState) -> str:
    return json.dumps(state.to_dict(), ensure_ascii=False)


class CatalogStateMachine:
    def __init__(self, catalog: TocCatalogIndex) -> None:
        self._catalog = catalog

    def apply_user_turn(
        self,
        *,
        question: str,
        state: CatalogDialogState,
    ) -> tuple[CatalogDialogState, Optional[CatalogNode], str]:
        """
        返回 (新状态, 当前锚定节点, 动作)。
        动作: reset | anchor | stay
        """
        if self._catalog.is_reset_intent(question):
            return CatalogDialogState(), None, "reset"

        current = self._catalog.get(state.node_uuid) if state.node_uuid else None
        matched = self._catalog.match_node(question, current=current, prefer_subtree=True)

        if matched and self._catalog.can_advance(current, matched):
            level = self._catalog.dialog_level(matched)
            new_state = CatalogDialogState(
                node_uuid=matched.uuid,
                path_titles=list(matched.path_titles),
                root_guide_shown=True if level > 0 else state.root_guide_shown,
                dialog_level=level,
            )
            return new_state, matched, "anchor"

        if current:
            return state, current, "stay"

        if matched:
            level = self._catalog.dialog_level(matched)
            new_state = CatalogDialogState(
                node_uuid=matched.uuid,
                path_titles=list(matched.path_titles),
                root_guide_shown=level > 0,
                dialog_level=level,
            )
            return new_state, matched, "anchor"

        return state, current, "stay"

    def should_show_root_guide(self, state: CatalogDialogState) -> bool:
        return not state.root_guide_shown and state.dialog_level <= 0

    def guide_candidates(self, state: CatalogDialogState, node: Optional[CatalogNode]) -> List[CatalogNode]:
        level = state.dialog_level
        if level <= 0 and not state.root_guide_shown:
            return self._catalog.roots[:6]
        if level <= 0:
            return self._catalog.roots[:6]
        if level == 1 and node:
            return self._catalog.children_of(node)[:6]
        if level == 2 and node:
            kids = self._catalog.children_of(node)
            if kids:
                return kids[:6]
            return self._catalog.related_in_catalog(node, limit=3)
        return []
