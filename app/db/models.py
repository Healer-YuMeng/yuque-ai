from __future__ import annotations

DOCUMENTS_DDL = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

CHUNKS_DDL = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    chunk_order INTEGER NOT NULL,
    snippet TEXT NOT NULL,
    FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
);
"""

QA_LOGS_DDL = """
CREATE TABLE IF NOT EXISTS qa_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    sources TEXT NOT NULL,
    fallback_used INTEGER NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

LEAD_CAPTURES_DDL = """
CREATE TABLE IF NOT EXISTS lead_captures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    contact_type TEXT NOT NULL,
    contact_value TEXT NOT NULL,
    visitor_type TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

LEAD_CAPTURES_UNIQUE_INDEX = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_lead_captures_session_contact
ON lead_captures(session_id, contact_type, contact_value);
"""

CHAT_SESSIONS_DDL = """
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id TEXT PRIMARY KEY,
    chat_mode TEXT NOT NULL,
    advisor_role TEXT NOT NULL DEFAULT 'sales',
    visitor_type TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

CHAT_MESSAGES_DDL = """
CREATE TABLE IF NOT EXISTS chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(session_id) REFERENCES chat_sessions(session_id)
);
"""

CHAT_MESSAGES_SESSION_CREATED_INDEX = """
CREATE INDEX IF NOT EXISTS idx_chat_messages_session_created
ON chat_messages(session_id, created_at);
"""

