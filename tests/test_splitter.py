from app.data.splitter import RecursiveTextSplitter


def test_splitter_creates_overlapping_chunks() -> None:
    splitter = RecursiveTextSplitter(chunk_size=10, chunk_overlap=2)
    chunks = splitter.split_document(
        doc_id="doc-1",
        title="标题",
        url="https://example.com/doc-1",
        text="abcdefghijklmnopqrstuvwxyz",
    )

    assert len(chunks) >= 3
    assert chunks[0].chunk_id == "doc-1:0"
    assert chunks[1].text.startswith("ijkl")

