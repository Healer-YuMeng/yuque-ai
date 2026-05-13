from __future__ import annotations

from app.conversation.contact_extractor import ContactHit, extract_contact
from app.conversation.visitor_intent import detect_intent_flags
from app.conversation.visitor_profile import VisitorType, detect_visitor_type


def build_visitor_generation_question(user_question: str) -> str:
    """
    在访客模式下把规则信号拼进「生成用问题」，供模型调整语气与留资节奏。
    检索仍应使用用户原句，避免关键词污染向量查询。
    """
    q = (user_question or "").strip()
    if not q:
        return q
    vt: VisitorType = detect_visitor_type(q)
    purchase, trial = detect_intent_flags(q)
    contact: ContactHit | None = extract_contact(q)

    lines: list[str] = []
    if vt != "unknown":
        lines.append(f"识别到的访客倾向（内部标签）: {vt}")
    if purchase:
        lines.append("意图线索: 购买/价格/合作相关")
    if trial:
        lines.append("意图线索: 试用/演示/体验相关")
    if contact:
        lines.append(f"用户本轮已提供联系方式: {contact.contact_type}={contact.value}")

    if not lines:
        return q

    return (
        "【以下段落为内部对话分析，请不要向用户复述本段标题或标签名；"
        "仅用于你调整讲解重点、语气与是否温和引导留资。】\n"
        + "\n".join(lines)
        + "\n\n用户原话：\n"
        + q
    )
