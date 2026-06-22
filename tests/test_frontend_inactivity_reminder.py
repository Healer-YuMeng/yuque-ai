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


def test_inactivity_reminder_does_not_replace_friend_v5_tags() -> None:
    app = (ROOT / "frontend/src/App.tsx").read_text(encoding="utf-8")
    assert "isInactivityReminderMessage" in app
    assert "tagDisplayAssistantId" in app
    assert "isInactivityReminder: true" in app
    assert "item.id === tagDisplayAssistantId" in app
    assert 'msg--inactivity' in app


def test_friend_v5_tags_render_short_labels() -> None:
    app = (ROOT / "frontend/src/App.tsx").read_text(encoding="utf-8")
    assert "function friendV5TagDisplayText(tag: string): string" in app
    assert "想了解一下价格？" in app
    assert "想看看优秀案例库？" in app
    assert "想申请测试账号，试试产品？" in app
    assert "friendV5TagDisplayText(tag)" in app
    assert "（${index + 1}）" not in app


def test_friend_v5_tag_click_still_uses_original_question_text() -> None:
    app = (ROOT / "frontend/src/App.tsx").read_text(encoding="utf-8")
    assert 'void askQuestion(tag, true, { triggerType: "tag", scene });' in app


def test_friend_v5_visitor_tags_use_smaller_font_than_body_copy() -> None:
    css = (ROOT / "frontend/src/index.css").read_text(encoding="utf-8")
    section = css.split(".app-shell--visitor .friend-v5-tag-chip {", 1)[1].split(
        ".app-shell--visitor .friend-v5-tag-chip::after {",
        1,
    )[0]
    assert "font-size: 15px;" in section
    assert "font-size: 17px;" not in section


def test_visitor_inactivity_prompt_uses_smaller_font() -> None:
    css = (ROOT / "frontend/src/index.css").read_text(encoding="utf-8")
    section = css.split(".app-shell--visitor .msg.assistant.msg--inactivity .bubble {", 1)[1].split(
        ".app-shell--visitor .msg.assistant .stream-stage--pending {",
        1,
    )[0]
    assert "font-size: 16px;" in section


def test_mobile_trial_apply_dialog_keeps_side_margins() -> None:
    css = (ROOT / "frontend/src/index.css").read_text(encoding="utf-8")
    mobile_section = css.split(".visitor-dialog-mask {", 2)[2]
    mobile_section = mobile_section.split(".visitor-workspace-drawer {", 1)[0]
    assert "padding: 12px;" in mobile_section
    assert "width: min(420px, calc(100vw - 24px));" in mobile_section
    assert "width: 100%;" not in mobile_section


def test_trial_apply_dialog_includes_optional_email_field() -> None:
    app = (ROOT / "frontend/src/App.tsx").read_text(encoding="utf-8")
    assert "const [trialApplyEmail, setTrialApplyEmail] = useState(\"\");" in app
    assert "邮箱（选填）" in app
    assert "请输入邮箱" in app
    assert "!trialApplyContact.trim()" in app


def test_admin_customers_table_includes_email_column() -> None:
    admin_app = (ROOT / "frontend/src/AdminApp.tsx").read_text(encoding="utf-8")
    assert "email: string;" in admin_app
    assert "<th>邮箱</th>" in admin_app
    assert "{customer.email || \"—\"}" in admin_app
