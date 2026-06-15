from __future__ import annotations

from app.service.friend_v5_tags import FriendV5TagStreamFilter, FriendV5TagParseResult, fallback_tags_for_scene


def test_tag_filter_hides_split_marker_from_stream() -> None:
    parser = FriendV5TagStreamFilter(scene="人工智能通识教育")

    assert parser.feed("小为先帮你看一下。\n[TA") == "小为先帮你看一下。\n"
    assert parser.feed("GS]\n想看课程例子？\n想了解适合年级？\n[END_TAGS]") == ""
    result = parser.finish()

    assert result.answer == "小为先帮你看一下。"
    assert result.tags[:2] == ["想看课程例子？", "想了解适合年级？"]
    assert "[TAGS]" not in result.answer


def test_tag_filter_strips_inline_tag_block() -> None:
    parser = FriendV5TagStreamFilter(scene="智能招生")

    visible = parser.feed("这块可以先看三个方向。[TAGS]招生线索怎么整理？,家长咨询怎么承接？[END_TAGS]")
    result = parser.finish()

    assert visible == "这块可以先看三个方向。"
    assert result.answer == "这块可以先看三个方向。"
    assert result.tags[0] == "招生线索怎么整理？"
    assert result.tags[1] == "家长咨询怎么承接？"


def test_tag_filter_fills_to_three_by_scene() -> None:
    parser = FriendV5TagStreamFilter(scene="跨学科项目化学习")

    assert parser.feed("可以先从主题和成果物看。[TAGS]想看项目案例？[END_TAGS]") == "可以先从主题和成果物看。"
    result = parser.finish()

    assert len(result.tags) == 3
    assert result.tags[0] == "想看项目案例？"
    assert any(tag in fallback_tags_for_scene("跨学科项目化学习") for tag in result.tags[1:])


def test_filter_hides_sources_block_and_extracts_urls() -> None:
    parser = FriendV5TagStreamFilter(scene="人工智能通识教育")

    visible = parser.feed(
        "这是正文[^1]更多[^2]\n"
        "[SOURCES]\n"
        "https://a.com/article\n"
        "https://b.com/news\n"
        "[/SOURCES]\n"
        "[TAGS]\n"
        "想看课程例子？\n"
        "想了解适合年级？\n"
        "[END_TAGS]"
    )
    result = parser.finish()

    assert result.answer == "这是正文更多"
    assert result.source_urls == ["https://a.com/article", "https://b.com/news"]
    assert result.tags[:2] == ["想看课程例子？", "想了解适合年级？"]
    assert visible == "这是正文[^1]更多[^2]\n\n"


def test_filter_sources_followed_by_tags_stream() -> None:
    parser = FriendV5TagStreamFilter(scene="智能招生")

    assert parser.feed("先看方向[SOURCES]\nhttps://x.com\n[/SOURCES]") == "先看方向"
    assert parser.feed("[TAGS]标签A\n标签B\n[END_TAGS]") == ""
    result = parser.finish()

    assert result.source_urls == ["https://x.com"]
    assert result.tags[:2] == ["标签A", "标签B"]


def test_filter_extracts_mixed_source_urls_and_ignores_end_marker() -> None:
    parser = FriendV5TagStreamFilter(scene="智能招生")

    visible = parser.feed(
        "先看方向[SOURCES]\n"
        "www.youweiai.com/page1 https://school.example.edu/news [/SOURCES]\n"
        "[TAGS]标签A\n标签B\n[END_TAGS]"
    )
    result = parser.finish()

    assert visible == "先看方向\n"
    assert result.answer == "先看方向"
    assert result.source_urls == ["https://www.youweiai.com/page1", "https://school.example.edu/news"]


def test_filter_strips_bare_sources_marker_fragment() -> None:
    parser = FriendV5TagStreamFilter(scene="智能招生")

    parser.feed("产品价格这块可以先看三个方向。SOURCES]")
    result = parser.finish()

    assert result.answer == "产品价格这块可以先看三个方向。"
    assert "SOURCES" not in result.answer


def test_filter_strips_partial_sources_marker_on_finish() -> None:
    parser = FriendV5TagStreamFilter(scene="智能招生")

    assert parser.feed("先看报价单。[S") == "先看报价单。"
    result = parser.finish()

    assert result.answer == "先看报价单。"
    assert "[S" not in result.answer


def test_filter_strips_bare_s_bracket_fragment() -> None:
    parser = FriendV5TagStreamFilter(scene="智能招生")

    assert parser.feed("先看报价单。S]") == "先看报价单。"
    result = parser.finish()

    assert result.answer == "先看报价单。"
    assert "S]" not in result.answer


def test_filter_strips_broken_protocol_source_line() -> None:
    parser = FriendV5TagStreamFilter(scene="智能招生")

    parser.feed("先看三个方向。\n\n://www.yuque.com/example/doc\n\n需要继续吗？")
    result = parser.finish()

    assert result.answer == "先看三个方向。\n\n需要继续吗？"
    assert "://" not in result.answer


def test_filter_drops_placeholder_example_sources() -> None:
    parser = FriendV5TagStreamFilter(scene="智能招生")

    parser.feed(
        "先看方向[SOURCES]\n"
        "example.com/page1 https://example.com/page2 [/SOURCES]\n"
        "[TAGS]标签A\n标签B\n[END_TAGS]"
    )
    result = parser.finish()

    assert result.source_urls == []
    assert "example.com" not in result.answer


def test_filter_strips_trial_account_disclosure() -> None:
    parser = FriendV5TagStreamFilter(scene="人工智能通识教育")

    parser.feed("信息校验通过，已为您分配测试账号。\n\n【测试账号】\n账号：demo01\n密码：pass123")
    result = parser.finish()

    assert result.answer == "提交成功，我们会尽快与您联系。"
    assert "【测试账号】" not in result.answer
    assert "demo01" not in result.answer
    assert "pass123" not in result.answer


def test_filter_fixes_ideas_pbl_typo() -> None:
    parser = FriendV5TagStreamFilter(scene="跨学科项目化学习")

    parser.feed("跨学科项目化学习（IDAS PBL）是由有为云联合上海师范大学打造的 AI 原生应用。")
    result = parser.finish()

    assert "IDEAS-PBL" in result.answer
    assert "IDAS PBL" not in result.answer
    assert "IDAS-PBL" not in result.answer


def test_filter_fixes_apple_steam_typo() -> None:
    parser = FriendV5TagStreamFilter(scene="人工智能通识教育")

    parser.feed("2. **苹果 STAM**：基于 Swift Playgrounds，融合编程与设计。\n如果您对苹果STAM课程也感兴趣。")
    result = parser.finish()

    assert "苹果 STEAM" in result.answer
    assert "苹果 STEAM课程" in result.answer
    assert "苹果 STAM" not in result.answer
    assert "苹果STAM" not in result.answer
