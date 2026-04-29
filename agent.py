from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from yuque_client import YuqueClient


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name, default) or "").strip()

def _build_toc_children_map(nodes: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    children: Dict[str, List[Dict[str, Any]]] = {}
    for n in nodes:
        parent = str(n.get("parent_uuid") or "")
        children.setdefault(parent, []).append(n)
    # 保持目录里原始顺序（API 返回本身就是有序的），这里不再排序
    return children

def _find_toc_node_uuid_by_doc_id(nodes: List[Dict[str, Any]], doc_id: int) -> Optional[str]:
    for n in nodes:
        if int(n.get("doc_id") or 0) == int(doc_id):
            return str(n.get("uuid") or "") or None
    return None

def _subtree_from_uuid(
    *,
    children_map: Dict[str, List[Dict[str, Any]]],
    root_uuid: str,
    max_nodes: int = 80,
) -> List[Dict[str, Any]]:
    """
    从 TOC 节点 uuid 开始，做 DFS 拉取子树（不含 root 自身，只含子节点）。
    返回扁平列表，含 level/title/type/url/doc_id。
    """
    out: List[Dict[str, Any]] = []
    stack: List[Dict[str, Any]] = list(reversed(children_map.get(root_uuid, [])))
    while stack and len(out) < max_nodes:
        node = stack.pop()
        out.append(
            {
                "type": node.get("type"),
                "title": node.get("title"),
                "url": node.get("url"),
                "doc_id": node.get("doc_id"),
                "level": node.get("level"),
                "uuid": node.get("uuid"),
                "parent_uuid": node.get("parent_uuid"),
            }
        )
        uuid = str(node.get("uuid") or "")
        if uuid:
            kids = children_map.get(uuid, [])
            if kids:
                stack.extend(reversed(kids))
    return out


def build_agent(
    *,
    yuque: YuqueClient,
    scope: Optional[str],
    model: Optional[str] = None,
) -> Any:
    base_url = _env("OPENAI_BASE_URL")
    resolved_model = model or _env(
        "OPENAI_MODEL",
        "deepseek-chat" if ("deepseek" in base_url.lower()) else "gpt-4o-mini",
    )

    llm = ChatOpenAI(
        model=resolved_model,
        temperature=0,
        base_url=base_url or None,
    )

    @tool
    def yuque_search_docs(q: str) -> List[Dict[str, Any]]:
        """在语雀里搜索文档。输入：q(关键词)。输出：命中文档列表（含 title/url/book_id/doc_id/slug/summary）。"""
        hits = yuque.search_docs(q=q, scope=scope, page=1)
        out: List[Dict[str, Any]] = []
        for h in hits[:10]:
            out.append(
                {
                    "title": h.title,
                    "url": h.url,
                    "summary": h.summary,
                    "info": h.info,
                    "book_id": h.book_id,
                    "doc_id": h.doc_id,
                    "slug": h.slug,
                }
            )
        return out

    @tool
    def yuque_get_doc(book_id: int, id_or_slug: str) -> Dict[str, Any]:
        """拉取语雀文档正文。输入：book_id(知识库ID)、id_or_slug(文档id或slug)。输出：title/url/format/body。"""
        doc = yuque.get_doc(book_id=book_id, id_or_slug=id_or_slug)
        body = (doc.body or "").strip()
        if len(body) > 12000:
            body = body[:12000] + "\n\n[内容过长，已截断]"
        return {"title": doc.title, "url": doc.url, "format": doc.format, "body": body}

    @tool
    def yuque_get_doc_subtree(book_id: int, doc_id: int) -> Dict[str, Any]:
        """
        获取某篇文档在“知识库目录树（TOC）”中的子树（用于回答“子文档/目录结构”问题）。
        输入：book_id(知识库ID)、doc_id(文档ID)。
        输出：root_uuid + children(扁平列表，每项含 type/title/url/doc_id/level)。
        """
        toc = yuque.get_book_toc(book=book_id)
        nodes: List[Dict[str, Any]] = []
        for t in toc:
            nodes.append(
                {
                    "uuid": t.uuid,
                    "type": t.type,
                    "title": t.title,
                    "url": t.url,
                    "doc_id": t.doc_id,
                    "level": t.level,
                    "parent_uuid": t.parent_uuid,
                }
            )
        root_uuid = _find_toc_node_uuid_by_doc_id(nodes, doc_id=doc_id) or ""
        children_map = _build_toc_children_map(nodes)
        subtree = _subtree_from_uuid(children_map=children_map, root_uuid=root_uuid) if root_uuid else []
        return {"root_uuid": root_uuid, "children": subtree}

    @tool
    def yuque_get_toc() -> List[Dict[str, Any]]:
        """
        获取知识库目录（TOC）列表（用于 search 命中为 0 时的兜底：从标题里挑文档）。
        输入：无（使用启动时配置的 scope，即环境变量 YUQUE_SCOPE）。
        输出：TOC 节点列表（含 type/title/url/doc_id/level/uuid/parent_uuid）。
        """
        if not scope:
            return []
        toc = yuque.get_book_toc(book=(scope or "").strip().strip("/"))
        out: List[Dict[str, Any]] = []
        for t in toc:
            out.append(
                {
                    "uuid": t.uuid,
                    "type": t.type,
                    "title": t.title,
                    "url": t.url,
                    "doc_id": t.doc_id,
                    "level": t.level,
                    "parent_uuid": t.parent_uuid,
                }
            )
        return out

    system = SystemMessage(
        content=(
            "你是企业知识库问答助手。你可以调用下面工具：\n"
            "- yuque_search_docs：在语雀搜索可能相关的文档\n"
            "- yuque_get_doc：拉取文档正文\n"
            "- yuque_get_doc_subtree：获取文档在知识库目录树里的子树（目录结构/子文档）\n\n"
            "- yuque_get_toc：获取知识库目录（当搜索为0时从标题兜底；依赖已配置的 scope）\n\n"
            "规则：\n"
            "1) 回答任何问题前，先调用 yuque_search_docs 搜索。搜索时不要直接用整句提问，先提炼 2-6 个字的关键词组合。\n"
            "   - 例如“老师插话后，AI 的结果还会发出去吗？”优先用“最终状态机 老师先回复 任务作废”等关键词去搜。\n"
            "   - 如果搜索结果为 0，调用 yuque_get_toc() 获取目录标题，再从标题中选最相关文档去拉正文。\n"
            "2) 若问题是“内容问答”，对最相关的 1-3 篇文档调用 yuque_get_doc 拉取正文。\n"
            "3) 若问题是“目录/子文档/结构/章节层级”，先用 yuque_search_docs 找到目标文档的 book_id/doc_id，"
            "再调用 yuque_get_doc_subtree 获取真实目录结构后回答。\n"
            "4) 工具调用要收敛：最多搜索 2 次、最多拉正文 3 篇、最多拉目录 1 次、最多拉 TOC 1 次；拿到信息后立刻输出最终回答，不要循环调用。\n"
            "5) 最终回答必须用中文，并在末尾给出“引用”列表（每条包含 title + url）。\n"
            "6) 如果文档里没有答案，要明确说未找到，并说明你看了哪些文档（引用）。\n"
        )
    )

    return create_react_agent(
        llm,
        tools=[yuque_search_docs, yuque_get_doc, yuque_get_doc_subtree, yuque_get_toc],
        prompt=system,
    )

