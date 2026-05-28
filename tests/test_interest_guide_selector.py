from __future__ import annotations

from app.conversation.interest_guide_selector import InterestGuideSelector
from app.schemas.chat import GuideDocTitleNode


def test_interest_selector_picks_related_titles() -> None:
    toc = [
        GuideDocTitleNode(
            uuid="r1",
            title="使用指南",
            level=1,
            children=[
                GuideDocTitleNode(uuid="c1", title="课堂怎么用", level=2),
                GuideDocTitleNode(uuid="c2", title="备课与教学流程", level=2),
            ],
        ),
        GuideDocTitleNode(
            uuid="r2",
            title="案例与社区",
            level=1,
            children=[
                GuideDocTitleNode(uuid="c3", title="优秀案例库", level=2),
            ],
        ),
    ]
    sel = InterestGuideSelector()
    picks = sel.pick_top3(
        question="我想了解备课怎么做",
        toc_nodes=toc,
        interests={"备课": {"score": 3}},
        visitor_type="teacher",
    )
    titles = [p.title for p in picks]
    assert any("备课" in t for t in titles)

