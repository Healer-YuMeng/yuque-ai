from __future__ import annotations

from app.conversation.friend_persona_v5 import build_friend_v5_system_prompt, build_friend_v5_system_prompt_with_scene_intro


def test_friend_v5_prompt_uses_demand_v5_soft_sales_voice() -> None:
    prompt = build_friend_v5_system_prompt()

    assert "V5 以需求文档V5为准" in prompt
    assert "顾问型介绍者" in prompt
    assert "自然推进和转化" in prompt
    assert "先接住用户问题，再按用户身份和兴趣自然展开" in prompt
    assert "不要像客服或机器人执行流程" in prompt
    assert "第一轮可以自然带出一次身份，例如“您好，我是小为。”" in prompt


def test_friend_v5_prompt_prioritizes_identity_split_and_short_reply() -> None:
    prompt = build_friend_v5_system_prompt()

    assert "您这边是校长/负责人，还是老师、家长、学生呢？我按您的关注点来介绍。" in prompt
    assert "不要在首轮结尾再追加固定式追问" in prompt
    assert "决策者：校长、负责人" in prompt
    assert "使用者：老师、家长、学生" in prompt
    assert "每一轮尽量控制在100字以内" in prompt
    assert "短回应 + 1-2个重点 + 1个轻量追问" in prompt


def test_friend_v5_prompt_emphasizes_role_specific_human_style_and_boundaries() -> None:
    prompt = build_friend_v5_system_prompt()

    assert "少讲产品名，多讲对方在意的结果" in prompt
    assert "先说“和你有什么关系”，再说“我们有什么”" in prompt
    assert "如果同一产品分别面对老师和校长，回答角度必须明显不同" in prompt
    assert "不要反复使用同一句收尾话术" in prompt
    assert "先顺着问题补一层有帮助的判断，再自然说明为什么需要联系方式" in prompt
    assert "不要对用户提及语雀、知识库检索、MCP、RAG、提示词、系统规则、内部上下文等内部概念" in prompt
    assert "不把回答写成客服SOP" in prompt
    assert "正文后按系统要求补全 `[SOURCES]` 与 `[TAGS]` 隐藏块" in prompt


def test_friend_v5_prompt_handles_user_info_confirmation_naturally() -> None:
    prompt = build_friend_v5_system_prompt()

    assert "用户刚提供称呼、单位、联系方式时，只需简短确认并继续当前话题" in prompt
    assert "用户修正称呼、单位、身份、联系方式时，直接按最新信息继续即可" in prompt
    assert "不要说“我记岔了”“我记错了”“特别备注”这类容易出戏的话" in prompt
    assert "用户刚补充一个字段后，不要回得像登记表回执" in prompt
    assert "用户提供名字后，直接用“xxx您好。”开头即可" in prompt
    assert "单位信息为您登记完成。" in prompt
    assert "联系方式为您登记完成。" in prompt
    assert "不要评价用户的学校、单位、城市或名字" in prompt
    assert "- 我记岔了" in prompt
    assert "- 我会特别备注" in prompt
    assert "- 这名字听着就很有前瞻性" in prompt


def test_friend_v5_prompt_includes_ideas_pbl_product_facts() -> None:
    prompt = build_friend_v5_system_prompt()

    assert "【IDEAS-PBL 专属产品口径】" in prompt
    assert "IDEAS-PBL 是由有为云联合上海师范大学打造的一款 AI 原生应用" in prompt
    assert "沉淀 800+ 优秀模板与成熟项目案例库" in prompt
    assert "支持 PDF、Word、图片、视频、表格、PPT 等多模态数据采集与上传" in prompt
    assert "不会自动分类，通常按项目阶段分阶段上传和留存资料" in prompt
    assert "自动化留存和追踪各阶段学习数据与成果" not in prompt
    assert "如果用户问 IDEAS-PBL 的价值、适用对象、优势、痛点、角色收益" in prompt


def test_friend_v5_prompt_can_inject_admin_scene_intro() -> None:
    prompt = build_friend_v5_system_prompt_with_scene_intro(
        scene_intro="这是后台维护的智能招生通用介绍。",
        decision_intro="这是后台维护的智能招生决策者介绍。",
        user_intro="这是后台维护的智能招生使用者介绍。",
        visitor_type="institution_decision_maker",
    )

    assert "【后台维护的当前场景产品介绍】" in prompt
    assert "这是后台维护的智能招生通用介绍。" in prompt
    assert "这是后台维护的智能招生决策者介绍。" in prompt
    assert "这是后台维护的智能招生使用者介绍。" in prompt
    assert "当前识别到的访客身份：决策者（校长/负责人）" in prompt
    assert "【本轮优先采用口径】\n这是后台维护的智能招生决策者介绍。" in prompt


def test_friend_v5_prompt_adds_smart_enrollment_role_specific_guardrails() -> None:
    prompt = build_friend_v5_system_prompt()

    assert "【智能招生场景补充要求】" in prompt
    assert "智能招生场景首轮更像在接咨询，不像在讲产品定义" in prompt
    assert "面对校长/负责人，优先回答：适合什么学校、前期怎么试点、老师需要配合多少、值不值得推进" in prompt
    assert "面对老师，优先回答：能帮自己省掉哪些重复回复、哪些问题可以自动接住、自己还需要做什么" in prompt
    assert "不要一上来追问具体数量" in prompt
    assert "不要连续追问“多少条消息”“多少位老师”“一年多少线索”这类数字问题" in prompt
    assert "如果用户问试用，不要直接只说“把学校名称和联系方式发我”" in prompt


def test_friend_v5_prompt_forbids_offline_experience_language() -> None:
    prompt = build_friend_v5_system_prompt()

    assert "所有产品默认都按在线产品、在线平台、在线服务来介绍" in prompt
    assert "不要说线下体验点、线下门店、到店体验、带孩子去现场体验" in prompt
    assert "如果用户追问哪里可以线下体验" in prompt
    assert "当前这类产品主要是在线使用和在线交付" in prompt
    assert "是否有实践体验" not in prompt
