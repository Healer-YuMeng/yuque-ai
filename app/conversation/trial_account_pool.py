from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from app.core.config import settings


@dataclass(frozen=True)
class TrialAccount:
    username: str
    password: str
    label: str = ""


def load_trial_accounts() -> List[TrialAccount]:
    raw = (settings.trial_accounts_json or "").strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    out: List[TrialAccount] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        user = str(item.get("username") or item.get("user") or "").strip()
        pwd = str(item.get("password") or item.get("pass") or "").strip()
        if not user or not pwd:
            continue
        label = str(item.get("label") or "").strip()
        out.append(TrialAccount(username=user, password=pwd, label=label))
    return out


def allocate_trial_account(session_id: str, accounts: Optional[Sequence[TrialAccount]] = None) -> Optional[TrialAccount]:
    pool = list(accounts or load_trial_accounts())
    sid = (session_id or "").strip()
    if not sid or not pool:
        return None
    idx = zlib.crc32(sid.encode("utf-8")) % len(pool)
    return pool[idx]
