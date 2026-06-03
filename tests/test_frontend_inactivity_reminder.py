from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_visitor_inactivity_reminder_waits_120_seconds() -> None:
    visitor_sales = (ROOT / "frontend/src/visitorSales.ts").read_text(encoding="utf-8")
    assert "export const INACTIVITY_MS = 120_000;" in visitor_sales


def test_visitor_inactivity_reminder_requires_no_pending_reply() -> None:
    app = (ROOT / "frontend/src/App.tsx").read_text(encoding="utf-8")
    assert "window.setTimeout(() => {" in app
    assert "}, INACTIVITY_MS)" in app
    assert "if (isStreamingRef.current) return;" in app
    assert "if (questionRef.current.trim()) return;" in app
    assert 's.messages.some((m) => m.role === "user")' in app
    assert "s.inactivityPromptSent || s.contactCollected || s.declinedContact" in app
