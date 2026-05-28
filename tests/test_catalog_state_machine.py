from __future__ import annotations

from app.conversation.catalog_state_machine import CatalogDialogState, CatalogStateMachine
from app.conversation.toc_catalog import TocCatalogIndex


def _sample_raw() -> list[dict]:
    return [
        {"uuid": "r1", "title": "平台介绍", "level": 1, "parent_uuid": ""},
        {"uuid": "r2", "title": "使用指南", "level": 1, "parent_uuid": ""},
        {"uuid": "c1", "title": "人工智能通识课程", "level": 2, "parent_uuid": "r1"},
        {"uuid": "c2", "title": "跨学科项目式学习", "level": 2, "parent_uuid": "r1"},
        {"uuid": "t1", "title": "教师端", "level": 2, "parent_uuid": "r2"},
        {"uuid": "d1", "title": "作业管理", "level": 3, "parent_uuid": "t1"},
    ]


def test_fsm_advances_into_subtree_not_root() -> None:
    catalog = TocCatalogIndex(_sample_raw())
    fsm = CatalogStateMachine(catalog)
    state = CatalogDialogState()
    state, node, action = fsm.apply_user_turn(question="我想看平台介绍", state=state)
    assert action == "anchor"
    assert node and node.title == "平台介绍"
    assert state.dialog_level == 1

    state, node2, action2 = fsm.apply_user_turn(question="人工智能通识课", state=state)
    assert action2 == "anchor"
    assert node2 and "通识" in node2.title
    assert len(state.path_titles) >= 2


def test_fsm_reset_on_home_phrase() -> None:
    catalog = TocCatalogIndex(_sample_raw())
    fsm = CatalogStateMachine(catalog)
    state = CatalogDialogState(node_uuid="r1", path_titles=["平台介绍"], dialog_level=1, root_guide_shown=True)
    state, node, action = fsm.apply_user_turn(question="回到首页", state=state)
    assert action == "reset"
    assert node is None
    assert state.dialog_level == 0


def test_fsm_blocks_shallow_jump_without_reset() -> None:
    catalog = TocCatalogIndex(_sample_raw())
    fsm = CatalogStateMachine(catalog)
    deep = catalog.match_node("作业管理", current=None, prefer_subtree=False)
    assert deep
    state = CatalogDialogState(
        node_uuid=deep.uuid,
        path_titles=list(deep.path_titles),
        dialog_level=3,
        root_guide_shown=True,
    )
    state2, node, action = fsm.apply_user_turn(question="平台介绍", state=state)
    assert action == "stay"
    assert state2.node_uuid == state.node_uuid


def test_level3_no_root_guide_candidates() -> None:
    catalog = TocCatalogIndex(_sample_raw())
    fsm = CatalogStateMachine(catalog)
    node = catalog.match_node("作业管理", current=None, prefer_subtree=False)
    assert node
    state = CatalogDialogState(
        node_uuid=node.uuid,
        path_titles=list(node.path_titles),
        dialog_level=3,
        root_guide_shown=True,
    )
    assert fsm.guide_candidates(state, node) == []
