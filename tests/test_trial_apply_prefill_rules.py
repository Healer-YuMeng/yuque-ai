from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_trial_apply_prefill_rejects_generic_role_names() -> None:
    app = (ROOT / "frontend/src/App.tsx").read_text(encoding="utf-8")
    assert "INVALID_TRIAL_APPLY_NAME_PATTERNS" in app
    assert "培训机构|学校里|机构里" in app
    assert "(?:老师|教师|家长|学生|同学|校长|主任|负责人)" in app


def test_trial_apply_prefill_rejects_role_only_org_values() -> None:
    app = (ROOT / "frontend/src/App.tsx").read_text(encoding="utf-8")
    assert "sanitizeTrialApplyOrgCandidate" in app
