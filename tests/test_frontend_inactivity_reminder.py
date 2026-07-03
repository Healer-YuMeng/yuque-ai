from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_visitor_inactivity_reminder_waits_120_seconds() -> None:
    visitor_sales = (ROOT / "frontend/src/visitorSales.ts").read_text(encoding="utf-8")
    assert "export const INACTIVITY_MS = 120_000;" in visitor_sales
    assert "如果您现在不方便继续看，也可以先申请测试账号。我让顾问把测试账号发您，后续顾问会和您联系，您有空再慢慢看。" in visitor_sales


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
    assert "isInactivityReminderMessage(item)" in app


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
    assert "width: min(332px, calc(100vw - 32px));" in mobile_section
    assert "width: 100%;" not in mobile_section


def test_mobile_trial_apply_dialog_uses_compact_spacing() -> None:
    css = (ROOT / "frontend/src/index.css").read_text(encoding="utf-8")
    mobile_card = css.split(".visitor-dialog-card {", 2)[2].split(".visitor-workspace-drawer {", 1)[0]
    assert "padding: 14px 12px calc(12px + env(safe-area-inset-bottom, 0px));" in mobile_card
    mobile_form = css.split(".app-shell--visitor .visitor-dialog-field input,", 2)[2].split(
        ".visitor-mobile-scene-bar {",
        1,
    )[0]
    assert "font-size: 14px;" in mobile_form
    input_only = css.split(".app-shell--visitor .visitor-dialog-field input {", 1)[1].split(
        ".app-shell--visitor .visitor-dialog-actions {",
        1,
    )[0]
    assert "height: 44px;" in input_only


def test_mobile_scene_overlay_uses_compact_layout() -> None:
    css = (ROOT / "frontend/src/index.css").read_text(encoding="utf-8")
    overlay_section = css.split(".app-shell--visitor.app-shell--visitor-scene-open .visitor-scene-column {", 1)[1].split(
        ".app-shell--visitor.app-shell--visitor-scene-open .visitor-scene-column .focus-scene-sidebar {",
        1,
    )[0]
    assert "left: 16px;" in overlay_section
    assert "right: 16px;" in overlay_section
    assert "padding: 10px 10px 12px;" in overlay_section
    assert "box-sizing: border-box;" in overlay_section

    close_section = css.split(".visitor-scene-overlay-close {", 1)[1].split(".visitor-dialog-mask {", 1)[0]
    assert "position: absolute;" in close_section
    assert "top: 12px;" in close_section
    assert "right: 12px;" in close_section
    assert "width: 28px;" in close_section
    assert "height: 28px;" in close_section
    assert "margin-bottom: 0;" in close_section

    title_section = css.split(".app-shell--visitor.app-shell--visitor-scene-open .visitor-scene-column .focus-scene-title-card {", 1)[1].split(
        ".app-shell--visitor.app-shell--visitor-scene-open .visitor-scene-column .focus-scene-sidebar {",
        1,
    )[0]
    assert "padding: 0 44px 0 12px;" in title_section

    mobile_section = css.rsplit("@media (max-width: 640px) {", 1)[1].split(
        "/* ------ 管理后台移动端基础适配（/admin） ------ */",
        1,
    )[0]
    assert "grid-template-columns: 1fr;" in mobile_section
    assert "min-height: 54px;" in mobile_section
    assert "padding: 10px 12px;" in mobile_section


def test_trial_apply_dialog_includes_optional_email_field() -> None:
    app = (ROOT / "frontend/src/App.tsx").read_text(encoding="utf-8")
    assert "const [trialApplyEmail, setTrialApplyEmail] = useState(\"\");" in app
    assert "邮箱（选填）" in app
    assert "请输入邮箱" in app
    assert "!trialApplyContact.trim()" in app


def test_inactivity_reminder_shows_trial_apply_button() -> None:
    app = (ROOT / "frontend/src/App.tsx").read_text(encoding="utf-8")
    assert "const showInlineTrialApplyButton =" in app
    assert 'isInactivityReminderMessage(item) || (item.trialApplyAvailable && item.isFriendV5)' in app
    assert 'className="msg-inline-action"' in app
    assert "申请测试账号" in app


def test_v4_trial_apply_offer_uses_unified_contact_copy() -> None:
    app = (ROOT / "frontend/src/App.tsx").read_text(encoding="utf-8")
    assert '"如果您现在不方便继续看，也可以先申请测试账号。我让顾问把测试账号发您，后续顾问会和您联系，您有空再慢慢看。"' in app
    assert "需要我给您提供这个模块的测试账号吗？" not in app


def test_friend_v5_can_also_show_trial_apply_button() -> None:
    app = (ROOT / "frontend/src/App.tsx").read_text(encoding="utf-8")
    assert "const showInlineTrialApplyButton =" in app
    assert "item.isFriendV5" in app
    assert "{showInlineTrialApplyButton ? (" in app
    assert 'className="msg-inline-action"' in app
    assert "friendTrialApplyAvailable || trialApplyIntentHit || hasUnifiedTrialApplyOffer" in app


def test_friend_v5_streaming_text_is_softened_before_done_payload() -> None:
    app = (ROOT / "frontend/src/App.tsx").read_text(encoding="utf-8")
    assert "function softenFriendV5StreamingText(text: string): string" in app
    assert 'out = out.replace(/^收到。/, "好的。");' in app
    assert 'out = out.replace("默认以腾讯青少年人工智能课程为主线，包含", "腾讯青少年人工智能课程是一套包含");' in app
    assert 'out = out.replace("这块默认先推腾讯青少年人工智能课程，是一套", "腾讯青少年人工智能课程是一套");' in app
    assert 'out = out.replace("这块默认先讲腾讯青少年人工智能课程，是一套", "腾讯青少年人工智能课程是一套");' in app
    assert 'out = out.replace("这块默认是腾讯青少年人工智能课程，提供", "腾讯青少年人工智能课程提供");' in app
    assert 'out = out.replace("这块主要推腾讯青少年人工智能课程，是一套", "腾讯青少年人工智能课程是一套");' in app
    assert "const nextText = item.isFriendV5 ? softenFriendV5StreamingText(item.text + token) : item.text + token;" in app


def test_trial_apply_intent_also_matches_try_product_phrase() -> None:
    app = (ROOT / "frontend/src/App.tsx").read_text(encoding="utf-8")
    assert "试一试(这个)?产品|体验(一下)?产品" in app


def test_mobile_visitor_composer_hides_placeholder_copy() -> None:
    app = (ROOT / "frontend/src/App.tsx").read_text(encoding="utf-8")
    assert "const [isMobileViewport, setIsMobileViewport] = useState(() =>" in app
    assert "window.innerWidth <= VISITOR_MOBILE_MAX_WIDTH" in app
    assert 'IS_VISITOR_ROUTE && isMobileViewport' in app
    assert '? ""' in app
    assert ': "请输入您的问题，我们会尽快与您对接..."' in app


def test_selected_focus_scene_button_is_disabled_to_prevent_repeat_click() -> None:
    app = (ROOT / "frontend/src/App.tsx").read_text(encoding="utf-8")
    assert "const isFocusSceneDisabled = (scene: FocusScene) =>" in app
    assert "Boolean(activeFocusScene && activeFocusScene === scene)" in app
    assert "disabled={isStreaming || isFocusSceneDisabled(scene)}" in app


def test_admin_customers_table_includes_email_column() -> None:
    admin_app = (ROOT / "frontend/src/AdminApp.tsx").read_text(encoding="utf-8")
    assert "email: string;" in admin_app
    assert "<th>邮箱</th>" in admin_app
    assert "{customer.email || \"—\"}" in admin_app
