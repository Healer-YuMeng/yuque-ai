from __future__ import annotations

from app.db.models import schema_statements


def test_chat_messages_recent_lookup_index_matches_query_order() -> None:
    sqlite_schema = "\n".join(schema_statements(dialect="sqlite"))
    postgres_schema = "\n".join(schema_statements(dialect="postgres"))

    assert "ON chat_messages(session_id, id DESC)" in sqlite_schema
    assert "ON chat_messages(session_id, id DESC)" in postgres_schema
