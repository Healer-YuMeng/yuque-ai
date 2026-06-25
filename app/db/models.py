from __future__ import annotations

from typing import List


def schema_statements(*, dialect: str) -> List[str]:
    if dialect == "postgres":
        return _postgres_schema_statements()
    return _sqlite_schema_statements()


def _sqlite_schema_statements() -> List[str]:
    return [
        """
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            url TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL,
            chunk_order INTEGER NOT NULL,
            snippet TEXT NOT NULL,
            FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS qa_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            sources TEXT NOT NULL,
            fallback_used INTEGER NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS lead_captures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            contact_type TEXT NOT NULL,
            contact_value TEXT NOT NULL,
            visitor_type TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_lead_captures_session_contact
        ON lead_captures(session_id, contact_type, contact_value);
        """,
        """
        CREATE TABLE IF NOT EXISTS chat_sessions (
            session_id TEXT PRIMARY KEY,
            chat_mode TEXT NOT NULL,
            advisor_role TEXT NOT NULL DEFAULT 'sales',
            visitor_type TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(session_id) REFERENCES chat_sessions(session_id)
        );
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_chat_messages_session_created
        ON chat_messages(session_id, created_at);
        """,
        """
        CREATE TABLE IF NOT EXISTS chat_session_profiles (
            session_id TEXT PRIMARY KEY,
            display_name TEXT,
            visitor_type TEXT,
            org_name TEXT,
            interests_json TEXT NOT NULL DEFAULT '{}',
            focused_doc_ids_json TEXT NOT NULL DEFAULT '[]',
            catalog_state_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS admin_video_assets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scene_key TEXT NOT NULL,
            scene_name TEXT NOT NULL,
            title TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            stored_filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_url TEXT NOT NULL,
            mime_type TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            duration_seconds INTEGER,
            status TEXT NOT NULL DEFAULT 'active',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_admin_video_assets_scene_status
        ON admin_video_assets(scene_key, status, created_at);
        """,
        """
        CREATE TABLE IF NOT EXISTS admin_scene_intros (
            scene_key TEXT PRIMARY KEY,
            scene_name TEXT NOT NULL,
            intro_text TEXT NOT NULL DEFAULT '',
            decision_intro_text TEXT NOT NULL DEFAULT '',
            user_intro_text TEXT NOT NULL DEFAULT '',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """,
    ]


def _postgres_schema_statements() -> List[str]:
    return [
        """
        CREATE EXTENSION IF NOT EXISTS vector;
        """,
        """
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            url TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
            chunk_order INTEGER NOT NULL,
            snippet TEXT NOT NULL,
            chunk_text TEXT,
            embedding vector
        );
        """,
        """
        ALTER TABLE chunks ADD COLUMN IF NOT EXISTS chunk_text TEXT;
        """,
        """
        ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding vector;
        """,
        """
        CREATE TABLE IF NOT EXISTS qa_logs (
            id BIGSERIAL PRIMARY KEY,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            sources TEXT NOT NULL,
            fallback_used BOOLEAN NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS lead_captures (
            id BIGSERIAL PRIMARY KEY,
            session_id TEXT NOT NULL,
            contact_type TEXT NOT NULL,
            contact_value TEXT NOT NULL,
            visitor_type TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_lead_captures_session_contact
        ON lead_captures(session_id, contact_type, contact_value);
        """,
        """
        CREATE TABLE IF NOT EXISTS chat_sessions (
            session_id TEXT PRIMARY KEY,
            chat_mode TEXT NOT NULL,
            advisor_role TEXT NOT NULL DEFAULT 'sales',
            visitor_type TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS chat_messages (
            id BIGSERIAL PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_chat_messages_session_created
        ON chat_messages(session_id, created_at);
        """,
        """
        CREATE TABLE IF NOT EXISTS chat_session_profiles (
            session_id TEXT PRIMARY KEY,
            display_name TEXT,
            visitor_type TEXT,
            org_name TEXT,
            interests_json TEXT NOT NULL DEFAULT '{}',
            focused_doc_ids_json TEXT NOT NULL DEFAULT '[]',
            catalog_state_json TEXT NOT NULL DEFAULT '{}',
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS admin_video_assets (
            id BIGSERIAL PRIMARY KEY,
            scene_key TEXT NOT NULL,
            scene_name TEXT NOT NULL,
            title TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            stored_filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_url TEXT NOT NULL,
            mime_type TEXT NOT NULL,
            file_size BIGINT NOT NULL,
            duration_seconds INTEGER,
            status TEXT NOT NULL DEFAULT 'active',
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_admin_video_assets_scene_status
        ON admin_video_assets(scene_key, status, created_at);
        """,
        """
        CREATE TABLE IF NOT EXISTS admin_scene_intros (
            scene_key TEXT PRIMARY KEY,
            scene_name TEXT NOT NULL,
            intro_text TEXT NOT NULL DEFAULT '',
            decision_intro_text TEXT NOT NULL DEFAULT '',
            user_intro_text TEXT NOT NULL DEFAULT '',
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """,
    ]
