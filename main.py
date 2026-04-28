from __future__ import annotations

import os
import sys
from typing import Optional

from dotenv import load_dotenv

from agent import build_agent
from yuque_client import YuqueAuthError, YuqueClient


def _best_effort_utf8_console() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name, default) or "").strip()


def _must_env(name: str) -> str:
    v = _env(name)
    if not v:
        raise RuntimeError(f"缺少环境变量：{name}")
    return v


def run_once(question: str, *, scope: Optional[str]) -> str:
    token = _must_env("YUQUE_TOKEN")
    yuque = YuqueClient(token=token)
    try:
        graph = build_agent(yuque=yuque, scope=scope)
        result = graph.invoke(
            {"messages": [("user", question)]},
            config={"recursion_limit": int(_env("GRAPH_RECURSION_LIMIT", "80") or 80)},
        )
        msgs = result.get("messages") or []
        return str(msgs[-1].content) if msgs else ""
    finally:
        yuque.close()


def main() -> int:
    _best_effort_utf8_console()
    load_dotenv()

    scope = _env("YUQUE_SCOPE") or None
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        print("用法：python main.py <你的问题>")
        print("示例：python main.py 退款多久到账？")
        return 2

    if not _env("OPENAI_API_KEY"):
        print('[缺少 OPENAI_API_KEY] 请先设置：setx OPENAI_API_KEY "你的OpenAI Key"')
        print("然后重新打开终端再运行。")
        return 4

    if not _env("YUQUE_TOKEN"):
        print('[缺少 YUQUE_TOKEN] 请先设置：setx YUQUE_TOKEN "你的语雀Token"')
        print('可选：setx YUQUE_SCOPE "团队login/知识库slug"（用于限定搜索范围）')
        print("然后重新打开终端再运行。")
        return 5

    try:
        answer = run_once(question, scope=scope)
        print(answer)
        return 0
    except YuqueAuthError as e:
        print(f"[语雀鉴权失败] {e}")
        print('请设置：setx YUQUE_TOKEN "你的语雀Token"  然后重新打开终端再运行')
        return 3
    except Exception as e:
        print(f"[运行失败] {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

