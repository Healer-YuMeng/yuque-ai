from __future__ import annotations

from app.data.mcp_client import MCPDocMeta, MCPTocNode
from app.rag.retriever import Retriever


def test_wants_rich_doc_inventory() -> None:
    assert Retriever._wants_rich_doc_inventory("给我展示知识库有哪些文档，结构层次分明")
    assert not Retriever._wants_rich_doc_inventory("展示知识库有哪些文档")
    # 含「图片」但问的是正文，不应只拉目录+清单（否则无 get_doc 正文）
    assert not Retriever._wants_rich_doc_inventory(
        "乐高人工智能课程里人工智能素养配图下面的文字是什么"
    )
    assert Retriever._wants_rich_doc_inventory("各文档图片数统计表，结构一目了然")


def test_wants_document_visual_content() -> None:
    assert Retriever._wants_document_visual_content("乐高人工智能课程我想知道里面有什么图片内容")
    assert Retriever._wants_document_visual_content("该文档里有哪些插图需要渲染到前端")
    assert not Retriever._wants_document_visual_content("各文档图片数统计表，结构一目了然")
    assert not Retriever._wants_document_visual_content("知识库有哪些文档带封面图")
    assert not Retriever._wants_document_visual_content("成大事三步法写了什么")


def test_doc_list_not_triggered_for_doc_visual_question() -> None:
    q = "乐高人工智能课程这个文档里面有什么图片"
    assert Retriever._wants_document_visual_content(q)
    assert not Retriever._is_doc_list_question(q)


def test_format_mcp_combined_inventory_context() -> None:
    docs = [
        MCPDocMeta("1", "父文档", "p", "http://x", word_count=505, image_count=0, doc_type="Doc", visible=True),
        MCPDocMeta("2", "子文档", "c", "http://y", word_count=None, body_length=1105, image_count=None),
    ]
    toc = [
        MCPTocNode("父文档", 0, "1", "p", visible=True),
        MCPTocNode("子文档", 1, "2", "c", visible=False),
    ]
    text = Retriever._format_mcp_combined_inventory_context(docs, toc)
    assert "【合并知识库清单" in text
    assert "父文档" in text
    assert "（不可见）" in text
    assert "| 1 |" in text
    assert "505" in text
    assert "1105" in text
