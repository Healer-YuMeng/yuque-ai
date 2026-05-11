from __future__ import annotations

from app.api.chat_api import _toc_nodes_for_query, _toc_node_to_doc_meta
from app.data.yuque_loader import YuqueTocNode


def test_toc_nodes_for_query_keeps_match_and_ancestors() -> None:
    nodes = [
        YuqueTocNode(uuid="root", type="TITLE", title="第一章", url="", doc_id=None, level=1, parent_uuid=""),
        YuqueTocNode(uuid="c1", type="DOC", title="子文档", url="https://yuque.com/o/r/subslug", doc_id=9, level=2, parent_uuid="root"),
    ]
    out = _toc_nodes_for_query(nodes, "子文")
    assert [n.uuid for n in out] == ["root", "c1"]


def test_toc_nodes_for_query_empty_returns_all() -> None:
    nodes = [
        YuqueTocNode(uuid="a", type="DOC", title="Only", url="", doc_id=1, level=1, parent_uuid=""),
    ]
    out = _toc_nodes_for_query(nodes, "")
    assert len(out) == 1


def test_toc_node_to_doc_meta_doc_vs_title() -> None:
    doc = _toc_node_to_doc_meta(
        YuqueTocNode(
            uuid="u1",
            type="DOC",
            title="正文",
            url="https://www.yuque.com/org/repo/my-slug",
            doc_id=42,
            level=2,
            parent_uuid="p",
        )
    )
    assert doc.toc_kind == "doc"
    assert doc.toc_selectable is True
    assert doc.slug == "my-slug"
    assert doc.id == 42

    title = _toc_node_to_doc_meta(
        YuqueTocNode(uuid="u2", type="TITLE", title="分组", url="", doc_id=None, level=1, parent_uuid="")
    )
    assert title.toc_kind == "title"
    assert title.toc_selectable is False
