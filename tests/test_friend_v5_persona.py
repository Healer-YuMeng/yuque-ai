from __future__ import annotations

from app.conversation.friend_persona_v5 import build_friend_v5_system_prompt


def test_friend_v5_prompt_uses_demand_v5_soft_sales_voice() -> None:
    prompt = build_friend_v5_system_prompt()

    assert "V5 以需求文档V5为准" in prompt
    assert "更像懂产品的朋友型介绍者" in prompt
    assert "推进留资与转化，但方式更柔和" in prompt
    assert "少一点命令式推进，多一点建议式表达" in prompt
    assert "称呼 -> 工作单位 -> 联系方式 -> 感兴趣产品" in prompt


def test_friend_v5_prompt_teaches_example_like_human_sentence_shape() -> None:
    prompt = build_friend_v5_system_prompt()

    assert "咱们的通识课主打" in prompt
    assert "乐高课程：结合传感器与电机" in prompt
    assert "方便告诉我怎么称呼您吗？" in prompt
    assert "您目前是在学校还是培训机构？我可以结合场景说得更贴近一些。" in prompt
    assert "方便留个微信或电话吗？" in prompt


def test_friend_v5_prompt_blocks_ai_summary_and_placeholder_leaks() -> None:
    prompt = build_friend_v5_system_prompt()

    assert "不要说“先说结论”" in prompt
    assert "不要说“我帮你整理好了”" in prompt
    assert "不要输出“更多正文”" in prompt
    assert "客服腔" in prompt
    assert "百科腔" in prompt
    assert "汇报腔" in prompt
    assert "正文... [^1] 更多正文... [^2]" not in prompt
