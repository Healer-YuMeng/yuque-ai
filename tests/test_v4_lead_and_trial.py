from __future__ import annotations

import pytest

from app.conversation.lead_nudge_policy import LeadNudgePolicy
from app.conversation.trial_account_pool import allocate_trial_account
from app.conversation.v4_lead_outreach import V4LeadOutreach
from app.core.config import settings
from app.db.repositories import ChatMessageRow


class _FakeLeadRepo:
    def __init__(self) -> None:
        self.rows: list = []

    async def try_insert_lead(self, **kwargs) -> bool:
        self.rows.append(kwargs)
        return True

    async def has_lead_for_session(self, *, session_id: str) -> bool:
        return any(r.get("session_id") == session_id for r in self.rows)


@pytest.mark.asyncio
async def test_v4_lead_detects_trial_intent() -> None:
    outreach = V4LeadOutreach(
        lead_policy=LeadNudgePolicy(rounds_threshold=99, stay_seconds_threshold=999),
        lead_capture_repository=_FakeLeadRepo(),
    )
    turn = await outreach.ingest_user_turn(
        session_id="s1",
        question="我想申请测试体验一下",
        profile=None,
        catalog_path=["平台介绍"],
    )
    assert turn.interests_patch.get("_lead", {}).get("wants_trial") is True


def test_trial_account_stable_per_session(monkeypatch: pytest.MonkeyPatch) -> None:
    accounts_json = '[{"username":"u1","password":"p1"},{"username":"u2","password":"p2"}]'
    object.__setattr__(settings, "trial_accounts_json", accounts_json)
    a1 = allocate_trial_account("session-abc")
    a2 = allocate_trial_account("session-abc")
    a3 = allocate_trial_account("session-xyz")
    assert a1 and a2
    assert a1.username == a2.username
    assert a1.username != a3.username
