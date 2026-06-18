from __future__ import annotations

import argparse
import asyncio
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.core.config import settings
from app.db.repositories import DocumentRepository
from app.db.session import DatabaseSessionFactory


RESET_TABLES = [
    "chat_messages",
    "lead_captures",
    "qa_logs",
    "chunks",
    "documents",
    "chat_sessions",
    "chat_session_profiles",
    "admin_video_assets",
    "admin_scene_intros",
]

SEQUENCE_COLUMNS = {
    "qa_logs": "id",
    "lead_captures": "id",
    "chat_messages": "id",
    "admin_video_assets": "id",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将现有 SQLite 数据迁移到 PostgreSQL。")
    parser.add_argument(
        "--sqlite-path",
        default=str(settings.sqlite_path),
        help="源 SQLite 文件路径，默认读取 app/core/config.py 里的 sqlite_path。",
    )
    parser.add_argument(
        "--database-url",
        default=settings.database_url,
        help="目标 PostgreSQL DATABASE_URL，默认读取 .env 中的 DATABASE_URL。",
    )
    return parser.parse_args()


def _connect_sqlite(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {str(row["name"]) for row in rows}


def _fetch_rows(conn: sqlite3.Connection, table_name: str, columns: Sequence[str]) -> list[tuple]:
    sql = f"SELECT {', '.join(columns)} FROM {table_name}"
    rows = conn.execute(sql).fetchall()
    return [tuple(row[col] for col in columns) for row in rows]


def _normalize_qa_logs(rows: Iterable[tuple]) -> list[tuple]:
    out = []
    for row in rows:
        item = list(row)
        if len(item) >= 5:
            item[4] = bool(item[4])
        out.append(tuple(item))
    return out


def _normalize_timestamp(value):
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    return datetime.strptime(text[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)


def _normalize_rows_for_postgres(*, columns: Sequence[str], rows: Iterable[tuple]) -> list[tuple]:
    normalized = []
    for row in rows:
        item = list(row)
        for idx, column in enumerate(columns):
            if column in {"created_at", "updated_at"}:
                item[idx] = _normalize_timestamp(item[idx])
        normalized.append(tuple(item))
    return normalized


async def _clear_target(conn) -> None:
    for table in RESET_TABLES:
        await conn.execute(f"DELETE FROM {table}")
    await conn.commit()


async def _copy_table(conn, *, table_name: str, columns: Sequence[str], rows: Iterable[tuple]) -> None:
    rows = _normalize_rows_for_postgres(columns=columns, rows=rows)
    if not rows:
        return
    placeholders = ", ".join("?" for _ in columns)
    sql = f"INSERT INTO {table_name}({', '.join(columns)}) VALUES ({placeholders})"
    await conn.executemany(sql, rows)
    await conn.commit()


async def _reset_sequences(conn) -> None:
    for table_name, column_name in SEQUENCE_COLUMNS.items():
        await conn.execute(
            "SELECT setval(pg_get_serial_sequence(?, ?), COALESCE((SELECT MAX(" + column_name + f") FROM {table_name}), 1), "
            "COALESCE((SELECT MAX(" + column_name + f") FROM {table_name}), NULL) IS NOT NULL)",
            (table_name, column_name),
        )
    await conn.commit()


async def migrate(sqlite_path: str, database_url: str) -> None:
    if not database_url.startswith(("postgres://", "postgresql://")):
        raise RuntimeError("DATABASE_URL 必须是 PostgreSQL 连接串。")

    sqlite_file = Path(sqlite_path)
    if not sqlite_file.exists():
        raise FileNotFoundError(f"SQLite 文件不存在：{sqlite_file}")

    session_factory = DatabaseSessionFactory(database_url)
    target_repository = DocumentRepository(session_factory)
    await target_repository.init_db()
    target_conn = await session_factory.connect()
    source_conn = _connect_sqlite(str(sqlite_file))

    try:
        await _clear_target(target_conn)

        if _table_exists(source_conn, "documents"):
            await _copy_table(
                target_conn,
                table_name="documents",
                columns=("doc_id", "title", "url", "content_hash", "updated_at"),
                rows=_fetch_rows(source_conn, "documents", ("doc_id", "title", "url", "content_hash", "updated_at")),
            )

        if _table_exists(source_conn, "chunks"):
            await _copy_table(
                target_conn,
                table_name="chunks",
                columns=("chunk_id", "doc_id", "chunk_order", "snippet"),
                rows=_fetch_rows(source_conn, "chunks", ("chunk_id", "doc_id", "chunk_order", "snippet")),
            )

        if _table_exists(source_conn, "qa_logs"):
            await _copy_table(
                target_conn,
                table_name="qa_logs",
                columns=("id", "question", "answer", "sources", "fallback_used", "created_at"),
                rows=_normalize_qa_logs(
                    _fetch_rows(source_conn, "qa_logs", ("id", "question", "answer", "sources", "fallback_used", "created_at"))
                ),
            )

        if _table_exists(source_conn, "lead_captures"):
            await _copy_table(
                target_conn,
                table_name="lead_captures",
                columns=("id", "session_id", "contact_type", "contact_value", "visitor_type", "created_at"),
                rows=_fetch_rows(
                    source_conn,
                    "lead_captures",
                    ("id", "session_id", "contact_type", "contact_value", "visitor_type", "created_at"),
                ),
            )

        if _table_exists(source_conn, "chat_sessions"):
            await _copy_table(
                target_conn,
                table_name="chat_sessions",
                columns=("session_id", "chat_mode", "advisor_role", "visitor_type", "created_at", "updated_at"),
                rows=_fetch_rows(
                    source_conn,
                    "chat_sessions",
                    ("session_id", "chat_mode", "advisor_role", "visitor_type", "created_at", "updated_at"),
                ),
            )

        if _table_exists(source_conn, "chat_messages"):
            await _copy_table(
                target_conn,
                table_name="chat_messages",
                columns=("id", "session_id", "role", "content", "created_at"),
                rows=_fetch_rows(source_conn, "chat_messages", ("id", "session_id", "role", "content", "created_at")),
            )

        if _table_exists(source_conn, "chat_session_profiles"):
            profile_columns = _table_columns(source_conn, "chat_session_profiles")
            base_rows = source_conn.execute("SELECT * FROM chat_session_profiles").fetchall()
            rows = []
            for row in base_rows:
                rows.append(
                    (
                        row["session_id"],
                        row["display_name"],
                        row["visitor_type"],
                        row["org_name"],
                        row["interests_json"],
                        row["focused_doc_ids_json"],
                        row["catalog_state_json"] if "catalog_state_json" in profile_columns else "{}",
                        row["created_at"],
                        row["updated_at"],
                    )
                )
            await _copy_table(
                target_conn,
                table_name="chat_session_profiles",
                columns=(
                    "session_id",
                    "display_name",
                    "visitor_type",
                    "org_name",
                    "interests_json",
                    "focused_doc_ids_json",
                    "catalog_state_json",
                    "created_at",
                    "updated_at",
                ),
                rows=rows,
            )

        if _table_exists(source_conn, "admin_video_assets"):
            await _copy_table(
                target_conn,
                table_name="admin_video_assets",
                columns=(
                    "id",
                    "scene_key",
                    "scene_name",
                    "title",
                    "original_filename",
                    "stored_filename",
                    "file_path",
                    "file_url",
                    "mime_type",
                    "file_size",
                    "duration_seconds",
                    "status",
                    "created_at",
                    "updated_at",
                ),
                rows=_fetch_rows(
                    source_conn,
                    "admin_video_assets",
                    (
                        "id",
                        "scene_key",
                        "scene_name",
                        "title",
                        "original_filename",
                        "stored_filename",
                        "file_path",
                        "file_url",
                        "mime_type",
                        "file_size",
                        "duration_seconds",
                        "status",
                        "created_at",
                        "updated_at",
                    ),
                ),
            )

        if _table_exists(source_conn, "admin_scene_intros"):
            intro_columns = _table_columns(source_conn, "admin_scene_intros")
            base_rows = source_conn.execute("SELECT * FROM admin_scene_intros").fetchall()
            rows = []
            for row in base_rows:
                rows.append(
                    (
                        row["scene_key"],
                        row["scene_name"],
                        row["intro_text"],
                        row["decision_intro_text"] if "decision_intro_text" in intro_columns else "",
                        row["user_intro_text"] if "user_intro_text" in intro_columns else "",
                        row["created_at"],
                        row["updated_at"],
                    )
                )
            await _copy_table(
                target_conn,
                table_name="admin_scene_intros",
                columns=(
                    "scene_key",
                    "scene_name",
                    "intro_text",
                    "decision_intro_text",
                    "user_intro_text",
                    "created_at",
                    "updated_at",
                ),
                rows=rows,
            )

        await _reset_sequences(target_conn)
    finally:
        source_conn.close()
        await target_conn.close()
        await session_factory.close()


def main() -> None:
    args = parse_args()
    asyncio.run(migrate(args.sqlite_path, args.database_url))


if __name__ == "__main__":
    main()
