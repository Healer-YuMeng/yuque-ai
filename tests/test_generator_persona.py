from __future__ import annotations

from app.rag.generator import DeepSeekGenerator


def test_qwen_visitor_sales_system_enforces_consultative_persona() -> None:
    system = DeepSeekGenerator._system_message([], visitor_sales=True, qwen_style=True)
    assert "资深解决方案顾问" in system
    assert "需求总结 → 专业判断 → 推荐建议" in system
    assert "我更建议" in system
    assert "不要说「A也可以、B也可以、看需求决定」" in system
    assert "主动采集的客户字段只有三项：称呼、工作单位、联系方式" in system
    assert "禁止直接问“手机号是多少/留个微信吧/请填写联系方式”" in system
