from app.db.session import rewrite_sqlite_placeholders


def test_rewrite_sqlite_placeholders_replaces_positional_params() -> None:
    result = rewrite_sqlite_placeholders(
        "SELECT * FROM chat_messages WHERE session_id=? AND role=? ORDER BY id DESC LIMIT ?"
    )

    assert result.sql == "SELECT * FROM chat_messages WHERE session_id=$1 AND role=$2 ORDER BY id DESC LIMIT $3"
    assert result.param_count == 3


def test_rewrite_sqlite_placeholders_keeps_question_mark_inside_string_literals() -> None:
    result = rewrite_sqlite_placeholders(
        "SELECT '?' AS literal, title FROM documents WHERE url LIKE ? AND note='what? now?'"
    )

    assert result.sql == "SELECT '?' AS literal, title FROM documents WHERE url LIKE $1 AND note='what? now?'"
    assert result.param_count == 1
