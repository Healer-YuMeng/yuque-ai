from __future__ import annotations

from app.core.datetime_util import sqlite_utc_to_cst


def test_sqlite_utc_to_cst_adds_eight_hours() -> None:
    assert sqlite_utc_to_cst("2026-06-17 01:45:44") == "2026-06-17 09:45:44"


def test_sqlite_utc_to_cst_handles_empty_and_invalid() -> None:
    assert sqlite_utc_to_cst("") == ""
    assert sqlite_utc_to_cst(None) == ""
    assert sqlite_utc_to_cst("not-a-date") == "not-a-date"
