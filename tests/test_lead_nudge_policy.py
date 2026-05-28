from __future__ import annotations

from datetime import datetime

from app.conversation.lead_nudge_policy import LeadNudgePolicy
from app.db.repositories import ChatMessageRow


def test_lead_nudge_rounds_triggered() -> None:
    policy = LeadNudgePolicy(rounds_threshold=5, stay_seconds_threshold=120)
    history = [ChatMessageRow(role="user", content=f"q{i}", created_at="2026-01-01 10:00:00") for i in range(4)]
    decision = policy.decide(
        question="请介绍一下价格",
        history=history,
        has_existing_lead=False,
        now=datetime(2026, 1, 1, 10, 0, 30),
    )
    assert decision.triggered is True
    assert decision.reason == "rounds"


def test_lead_nudge_stay_triggered() -> None:
    policy = LeadNudgePolicy(rounds_threshold=5, stay_seconds_threshold=120)
    history = [
        ChatMessageRow(role="assistant", content="欢迎", created_at="2026-01-01 10:00:00"),
        ChatMessageRow(role="user", content="你好", created_at="2026-01-01 10:00:10"),
    ]
    decision = policy.decide(
        question="平台适合什么场景",
        history=history,
        has_existing_lead=False,
        now=datetime(2026, 1, 1, 10, 2, 30),
    )
    assert decision.triggered is True
    assert decision.reason == "stay"


def test_lead_nudge_skips_when_has_lead() -> None:
    policy = LeadNudgePolicy(rounds_threshold=5, stay_seconds_threshold=120)
    decision = policy.decide(question="这是我的手机号 13812345678", history=[], has_existing_lead=True)
    assert decision.triggered is False
