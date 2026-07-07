from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_trial_apply_prefill_rejects_generic_role_names() -> None:
    app = (ROOT / "frontend/src/App.tsx").read_text(encoding="utf-8")
    assert "extractTrialApplyDraft" not in app
    assert "/visitor/profile" not in app


def test_trial_apply_prefill_rejects_role_only_org_values() -> None:
    app = (ROOT / "frontend/src/App.tsx").read_text(encoding="utf-8")
    assert "setTrialApplyName(\"\")" in app
    assert "setTrialApplyOrg(\"\")" in app
    assert "setTrialApplyContact(\"\")" in app
    assert "setTrialApplyEmail(\"\")" in app
