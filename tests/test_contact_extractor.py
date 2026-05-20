from __future__ import annotations

from app.conversation.contact_extractor import extract_contact, looks_like_decline_followup
from app.conversation.visitor_prompt import build_visitor_generation_question


def test_extract_phone() -> None:
    hit = extract_contact("我的电话是 13800138000")
    assert hit is not None
    assert hit.contact_type == "phone"
    assert hit.value == "13800138000"


def test_extract_wechat_label() -> None:
    hit = extract_contact("微信：edu_ai_2026")
    assert hit is not None
    assert hit.contact_type == "wechat"
    assert hit.value == "edu_ai_2026"


def test_decline_followup() -> None:
    assert looks_like_decline_followup("别联系我了") is True
    assert looks_like_decline_followup("想了解价格") is False


def test_build_visitor_generation_question_includes_signals() -> None:
    q = build_visitor_generation_question("我是老师，想试用一下")
    assert "teacher" in q or "老师" in q
    assert "用户原话" in q
