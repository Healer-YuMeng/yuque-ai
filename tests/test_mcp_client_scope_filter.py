from app.data.mcp_client import YuqueMCPClient


def test_extract_repo_from_full_url() -> None:
    got = YuqueMCPClient._extract_repo_from_url("https://www.yuque.com/suesun-yb1bi/sspenu/doc-1")
    assert got == "suesun-yb1bi/sspenu"


def test_extract_repo_from_relative_url() -> None:
    got = YuqueMCPClient._extract_repo_from_url("/suesun-yb1bi/sspenu/doc-1")
    assert got == "suesun-yb1bi/sspenu"


def test_extract_repo_from_plain_path() -> None:
    got = YuqueMCPClient._extract_repo_from_url("suesun-yb1bi/sspenu/doc-1")
    assert got == "suesun-yb1bi/sspenu"
