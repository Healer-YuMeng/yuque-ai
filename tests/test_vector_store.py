from app.storage.vector_store import format_pgvector, normalize_vector


def test_normalize_vector_returns_unit_length() -> None:
    result = normalize_vector([3.0, 4.0])
    assert abs((result[0] ** 2 + result[1] ** 2) ** 0.5 - 1.0) < 1e-6


def test_format_pgvector_serializes_normalized_values() -> None:
    assert format_pgvector([3.0, 4.0]) == "[0.60000000,0.80000000]"
