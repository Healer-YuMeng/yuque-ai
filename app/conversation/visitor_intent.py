from __future__ import annotations

from typing import FrozenSet


_PURCHASE: FrozenSet[str] = frozenset(
    (
        "购买",
        "买",
        "开通",
        "报价",
        "价格",
        "多少钱",
        "费用",
        "收费",
        "合作",
        "采购",
        "方案",
        "签约",
        "付款",
        "下单",
        "合同",
    )
)
_TRIAL: FrozenSet[str] = frozenset(
    (
        "试用",
        "体验",
        "demo",
        "演示",
        "演示安排",
        "预约",
        "讲解",
        "试试看",
        "申请",
        "联系销售",
        "销售",
    )
)


def detect_intent_flags(text: str) -> tuple[bool, bool]:
    """返回 (has_purchase_intent, has_trial_intent)。"""
    t = (text or "").strip()
    if not t:
        return False, False
    low = t.lower()
    purchase = any(k in t for k in _PURCHASE)
    trial = any(k in t for k in _TRIAL) or "demo" in low
    return purchase, trial
