from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Sequence

from app.db.repositories import ChatMessageRow


@dataclass(frozen=True)
class LeadNudgeDecision:
    triggered: bool
    reason: str = ""
    text: str = ""


class LeadNudgePolicy:
    def __init__(self, *, rounds_threshold: int, stay_seconds_threshold: int) -> None:
        self._rounds_threshold = max(1, int(rounds_threshold))
        self._stay_seconds_threshold = max(1, int(stay_seconds_threshold))

    def decide(
        self,
        *,
        question: str,
        history: Sequence[ChatMessageRow],
        has_existing_lead: bool,
        now: Optional[datetime] = None,
    ) -> LeadNudgeDecision:
        if has_existing_lead:
            return LeadNudgeDecision(triggered=False)

        if self._has_contact_hint(question):
            return LeadNudgeDecision(triggered=False)

        # 历史中已触发过留资引导时，整场不再重复
        if self._already_nudged(history):
            return LeadNudgeDecision(triggered=False)

        user_rounds = sum(1 for row in history if row.role == "user") + 1
        if user_rounds >= self._rounds_threshold:
            return LeadNudgeDecision(
                triggered=True,
                reason="rounds",
                text="如果您愿意，我可以安排顾问按您的场景给一版建议方案。留下电话或微信即可快速对接。",
            )

        last_assistant_time = self._latest_assistant_time(history)
        ref = now or datetime.now()
        if last_assistant_time is not None:
            delta_seconds = (ref - last_assistant_time).total_seconds()
            if delta_seconds >= self._stay_seconds_threshold:
                return LeadNudgeDecision(
                    triggered=True,
                    reason="stay",
                    text="您若希望继续深入了解，也可以留下联系方式，我这边让顾问结合您的需求跟进。",
                )
        return LeadNudgeDecision(triggered=False)

    @staticmethod
    def _latest_assistant_time(history: Sequence[ChatMessageRow]) -> Optional[datetime]:
        for row in reversed(history):
            if row.role != "assistant":
                continue
            ts = (row.created_at or "").strip()
            if not ts:
                continue
            parsed = LeadNudgePolicy._parse_time(ts)
            if parsed is not None:
                return parsed
        return None

    @staticmethod
    def _parse_time(raw: str) -> Optional[datetime]:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(raw, fmt)
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            return None

    @staticmethod
    def _already_nudged(history: Sequence[ChatMessageRow]) -> bool:
        """历史助理消息中已出现过留资/试用引导话术，则不再触发。"""
        nudge_markers = ("可以留一下", "留下联系方式", "留个电话", "开通试用账号", "申请测试账号", "安排顾问")
        for row in history:
            if row.role != "assistant":
                continue
            content = (row.content or "").strip()
            if any(m in content for m in nudge_markers):
                return True
        return False

    @staticmethod
    def _has_contact_hint(question: str) -> bool:
        q = (question or "").strip().lower()
        if not q:
            return False
        keys = ("电话", "手机号", "微信", "vx", "v:", "weixin", "@", "联系我")
        return any(k in q for k in keys)
