from __future__ import annotations

from app.data.yuque_loader import strip_yuque_leaks_from_text
from app.service.friend_dialog_orchestrator_v5 import _strip_inline_urls
from app.service.friend_v5_tags import _clean_answer


def test_strip_yuque_leaks_removes_bare_repo_path() -> None:
    text = "具体大纲我放在下面了。suesun-yb1bi/sspenu/tuv9fxvc39knpt7c 您可以先扫一眼。"
    assert strip_yuque_leaks_from_text(text) == "具体大纲我放在下面了。 您可以先扫一眼。"


def test_strip_yuque_leaks_removes_domain_url() -> None:
    text = "详见 https://www.yuque.com/suesun-yb1bi/sspenu/tuv9fxvc39knpt7c?singleDoc 这份资料。"
    assert strip_yuque_leaks_from_text(text) == "详见 这份资料。"


def test_strip_yuque_leaks_removes_yuque_domain_without_protocol() -> None:
    text = "参考 yuque.com/suesun-yb1bi/sspenu/tuv9fxvc39knpt7c 即可。"
    assert "yuque.com" not in strip_yuque_leaks_from_text(text)
    assert "sspenu" not in strip_yuque_leaks_from_text(text)


def test_strip_inline_urls_removes_bare_path_in_v5_answer() -> None:
    answer = _strip_inline_urls("请看 suesun-yb1bi/sspenu/tuv9fxvc39knpt7c 这份大纲。")
    assert "sspenu" not in answer
    assert answer == "请看 这份大纲。"


def test_strip_yuque_leaks_removes_trailing_owner_slug() -> None:
    text = "您可以先看看下方的课程指南，了解具体的模块安排。suesun-yb1bi."
    assert strip_yuque_leaks_from_text(text) == "您可以先看看下方的课程指南，了解具体的模块安排。"


def test_strip_yuque_leaks_removes_bare_repo_pair() -> None:
    text = "详见 suesun-yb1bi/sspenu 这份资料。"
    assert strip_yuque_leaks_from_text(text) == "详见 这份资料。"


def test_strip_yuque_leaks_keeps_ideas_pbl_product_name() -> None:
    text = "跨学科项目化学习（IDEAS-PBL）是由有为云联合上海师范大学打造的 AI 原生应用。"
    assert strip_yuque_leaks_from_text(text) == text


def test_clean_answer_removes_bare_path_from_streamed_body() -> None:
    answer = _clean_answer("正文如下 suesun-yb1bi/sspenu/tuv9fxvc39knpt7c\n\n[SOURCES]\n[/SOURCES]")
    assert "sspenu" not in answer
