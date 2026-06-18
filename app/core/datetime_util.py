from __future__ import annotations

from datetime import datetime, timedelta, timezone

_CST = timezone(timedelta(hours=8))
_SQLITE_DT_FMT = "%Y-%m-%d %H:%M:%S"


def sqlite_utc_to_cst(value: str | None) -> str:
    """将 SQLite CURRENT_TIMESTAMP（UTC）转为北京时间展示字符串。"""
    raw = (value or "").strip()
    if not raw:
        return ""
    text = raw[:19]
    try:
        dt_utc = datetime.strptime(text, _SQLITE_DT_FMT).replace(tzinfo=timezone.utc)
    except ValueError:
        return raw
    return dt_utc.astimezone(_CST).strftime(_SQLITE_DT_FMT)
