"""
Audit log — append-only SQLite store.

Uses SQLite for portability (no external service needed).
The INSERT-only invariant is enforced at the application layer:
no UPDATE or DELETE is ever issued against these tables.

Tables
------
submission_events   one row per POST /submit classification
appeal_events       one row per POST /appeal
"""

import json
import sqlite3
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "audit.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist. Safe to call on every startup."""
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS submission_events (
                log_id               TEXT PRIMARY KEY,
                content_id           TEXT NOT NULL,
                signal_1_score       REAL,
                signal_1_rationale   TEXT,
                signal_1_flags       TEXT,          -- JSON array
                signal_2_score       REAL,
                signal_2_label       TEXT,
                signal_2_features    TEXT,          -- JSON object
                combined_score       REAL,
                category             TEXT,
                contributing_features TEXT,         -- JSON array
                label_text           TEXT,
                created_at           TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS appeal_events (
                log_id               TEXT PRIMARY KEY,
                appeal_id            TEXT NOT NULL,
                content_id           TEXT NOT NULL,
                creator_statement    TEXT,
                status               TEXT NOT NULL,
                created_at           TEXT NOT NULL
            );
        """)
    logger.info("Audit DB initialised at %s", DB_PATH)


def log_submission(
    content_id: str,
    signal1: dict,
    signal2: dict,
    scoring,          # ScoringResult
    label_text: str,
) -> str:
    """Insert a submission classification event. Returns log_id."""
    log_id = str(uuid.uuid4())
    now = _now()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO submission_events
            (log_id, content_id, signal_1_score, signal_1_rationale,
             signal_1_flags, signal_2_score, signal_2_label, signal_2_features,
             combined_score, category, contributing_features, label_text, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                log_id,
                content_id,
                signal1.get("score"),
                signal1.get("rationale"),
                json.dumps(signal1.get("flags", [])),
                signal2.get("score"),
                signal2.get("label"),
                json.dumps(signal2.get("features", {})),
                scoring.combined_score,
                scoring.category,
                json.dumps(scoring.contributing_features),
                label_text,
                now,
            ),
        )
    logger.info("submission logged log_id=%s content_id=%s", log_id, content_id)
    return log_id


def log_appeal(content_id: str, creator_statement: str) -> tuple[str, str]:
    """Insert an appeal event. Returns (appeal_id, log_id)."""
    appeal_id = str(uuid.uuid4())
    log_id = str(uuid.uuid4())
    now = _now()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO appeal_events
            (log_id, appeal_id, content_id, creator_statement, status, created_at)
            VALUES (?,?,?,?,?,?)
            """,
            (log_id, appeal_id, content_id, creator_statement, "pending_review", now),
        )
    logger.info("appeal logged appeal_id=%s content_id=%s", appeal_id, content_id)
    return appeal_id, log_id


def get_submission(content_id: str) -> dict | None:
    """Fetch the most recent submission event for a content_id."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM submission_events WHERE content_id = ? ORDER BY created_at DESC LIMIT 1",
            (content_id,),
        ).fetchone()
    if row is None:
        return None
    d = dict(row)
    for key in ("signal_1_flags", "signal_2_features", "contributing_features"):
        if d.get(key):
            d[key] = json.loads(d[key])
    return d


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
