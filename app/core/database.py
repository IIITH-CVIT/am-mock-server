import json
import os
import sqlite3
from contextlib import contextmanager

from app.core.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS registrations (
    id TEXT PRIMARY KEY,
    full_name TEXT NOT NULL,
    date_of_visit TEXT,
    timeslot TEXT,
    ticket_category TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS vectors (
    id TEXT PRIMARY KEY,
    registration_id TEXT NOT NULL REFERENCES registrations(id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK (kind IN ('face', 'fingerprint')),
    source TEXT NOT NULL DEFAULT 'registration',
    model TEXT,
    dim INTEGER NOT NULL,
    vector TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_vectors_registration_id ON vectors(registration_id);
CREATE INDEX IF NOT EXISTS idx_vectors_kind ON vectors(kind);
"""


def init_db() -> None:
    os.makedirs(os.path.dirname(settings.database.path), exist_ok=True)
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        # Backfill columns for databases created before these fields existed.
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(registrations)")}
        for column in ("date_of_visit", "timeslot", "ticket_category"):
            if column not in existing:
                conn.execute(f"ALTER TABLE registrations ADD COLUMN {column} TEXT")


@contextmanager
def get_conn():
    conn = sqlite3.connect(settings.database.path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def insert_registration(
    conn: sqlite3.Connection,
    registration_id: str,
    full_name: str,
    date_of_visit: str,
    timeslot: str,
    ticket_category: str,
) -> None:
    conn.execute(
        "INSERT INTO registrations (id, full_name, date_of_visit, timeslot, ticket_category) "
        "VALUES (?, ?, ?, ?, ?)",
        (registration_id, full_name, date_of_visit, timeslot, ticket_category),
    )


def insert_vector(
    conn: sqlite3.Connection,
    vector_id: str,
    registration_id: str,
    kind: str,
    model: str,
    vector: list[float],
    source: str = "registration",
) -> None:
    conn.execute(
        "INSERT INTO vectors (id, registration_id, kind, source, model, dim, vector) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (vector_id, registration_id, kind, source, model, len(vector), json.dumps(vector)),
    )


def get_registration(conn: sqlite3.Connection, registration_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM registrations WHERE id = ?", (registration_id,)
    ).fetchone()


def list_registrations(conn: sqlite3.Connection, skip: int = 0, limit: int = 100) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM registrations ORDER BY created_at DESC LIMIT ? OFFSET ?",
        (limit, skip),
    ).fetchall()


def list_vectors_by_kind(
    conn: sqlite3.Connection, kind: str, dim: int, source: str | None = None
) -> list[sqlite3.Row]:
    if source:
        return conn.execute(
            "SELECT * FROM vectors WHERE kind = ? AND dim = ? AND source = ?",
            (kind, dim, source),
        ).fetchall()
    return conn.execute(
        "SELECT * FROM vectors WHERE kind = ? AND dim = ?", (kind, dim)
    ).fetchall()
