from __future__ import annotations

from app.core.config import settings

PRIMARY = "primary"
SECONDARY = "secondary"


def normalize_yuque_token_profile(raw: str | None) -> str:
    p = (raw or "").strip().lower()
    if p in ("secondary", "second", "b", "alt", "2"):
        return SECONDARY
    return PRIMARY


def yuque_token_for_profile(profile: str | None) -> str:
    if normalize_yuque_token_profile(profile) == SECONDARY:
        return (settings.yuque_token_secondary or "").strip()
    return (settings.yuque_token or "").strip()


def default_yuque_scope_for_profile(profile: str | None) -> str:
    if normalize_yuque_token_profile(profile) == SECONDARY:
        sec = (settings.yuque_scope_secondary or "").strip()
        return sec or (settings.yuque_scope or "").strip()
    return (settings.yuque_scope or "").strip()


def secondary_yuque_configured() -> bool:
    return bool((settings.yuque_token_secondary or "").strip())
