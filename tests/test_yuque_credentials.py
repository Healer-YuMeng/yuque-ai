from __future__ import annotations

from app.core.yuque_credentials import PRIMARY, SECONDARY, normalize_yuque_token_profile


def test_normalize_yuque_token_profile_defaults_to_primary() -> None:
    assert normalize_yuque_token_profile(None) == PRIMARY
    assert normalize_yuque_token_profile("") == PRIMARY
    assert normalize_yuque_token_profile("  ") == PRIMARY
    assert normalize_yuque_token_profile("primary") == PRIMARY


def test_normalize_yuque_token_profile_secondary_aliases() -> None:
    assert normalize_yuque_token_profile("secondary") == SECONDARY
    assert normalize_yuque_token_profile("SECONDARY") == SECONDARY
    assert normalize_yuque_token_profile("b") == SECONDARY
    assert normalize_yuque_token_profile("2") == SECONDARY
