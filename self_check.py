from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

from yuque_client import YuqueAuthError, YuqueApiError, YuqueClient


def _best_effort_utf8_console() -> None:
    # Windows 默认控制台编码可能导致中文乱码；尽量切到 UTF-8，不行就忽略。
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass


def main() -> int:
    _best_effort_utf8_console()
    load_dotenv()

    token = os.getenv("YUQUE_TOKEN", "").strip()
    scope = os.getenv("YUQUE_SCOPE", "").strip() or None
    q = os.getenv("YUQUE_TEST_QUERY", "").strip() or "退款"

    try:
        client = YuqueClient(token=token)
    except YuqueAuthError as e:
        print(f"[缺少 Token] {e}")
        print("请在当前 shell 设置：")
        print('  setx YUQUE_TOKEN "你的语雀Token"')
        print('  setx YUQUE_SCOPE "团队login/知识库slug"  （可选，用于限定搜索范围）')
        print('  setx YUQUE_TEST_QUERY "测试关键词"        （可选）')
        print("然后重新打开终端再运行：python self_check.py")
        return 2

    try:
        msg = client.hello()
        print(f"[OK] /hello: {msg!r}")

        hits = client.search_docs(q=q, scope=scope, page=1)
        print(f"[OK] /search: hits={len(hits)} scope={scope!r} q={q!r}")
        if not hits:
            print("未搜到文档。你可以换一个关键词，或检查 scope 是否写对。")
            return 0

        first = hits[0]
        book_id = first.book_id
        doc_id = first.doc_id
        slug = first.slug
        print(
            "[INFO] top1:",
            {"title": first.title, "url": first.url, "book_id": book_id, "doc_id": doc_id, "slug": slug},
        )

        if not book_id:
            print("search 返回里没有 book_id，无法自动拉取正文。你可以改用命中文档的 book_id 手动测试。")
            return 0

        id_or_slug = str(doc_id) if doc_id else (slug or "")
        if not id_or_slug:
            print("search 命中缺少 doc_id/slug，无法自动拉取正文。")
            return 0

        doc = client.get_doc(book_id=book_id, id_or_slug=id_or_slug)
        snippet = (doc.body or "").strip().replace("\n", " ")[:180]
        print(f"[OK] /repos/:book_id/docs/:id: title={doc.title!r} url={doc.url!r}")
        print(f"[OK] body_snippet: {snippet}...")
        return 0
    except (YuqueApiError, YuqueAuthError) as e:
        print(f"[FAIL] {e}")
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())

