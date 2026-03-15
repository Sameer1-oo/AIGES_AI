"""
AEGIS AI - Database Utilities
SQLite integration for media scan results and blocked links.
"""

import sqlite3
import os
from datetime import datetime

# ─── DB Path ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "registry", "block_registry.db")


# ─── Initialize Database ────────────────────────────────────────────────────
def init_db():
    """Create tables if they don't exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scanned_media (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            filename    TEXT    NOT NULL,
            type        TEXT    NOT NULL,
            result      TEXT    NOT NULL,
            confidence  REAL    NOT NULL,
            timestamp   TEXT    NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS blocked_links (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            url         TEXT    NOT NULL,
            status      TEXT    NOT NULL,
            timestamp   TEXT    NOT NULL
        )
    """)

    conn.commit()
    conn.close()


# ─── Save Media Scan ─────────────────────────────────────────────────────────
def save_media(filename: str, media_type: str, result: str, confidence: float):
    """Insert a scanned media record into the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO scanned_media (filename, type, result, confidence, timestamp)
        VALUES (?, ?, ?, ?, ?)
        """,
        (filename, media_type, result, confidence, datetime.utcnow().isoformat()),
    )
    conn.commit()
    last_id = cursor.lastrowid
    conn.close()
    return last_id


# ─── Save Link Scan ──────────────────────────────────────────────────────────
def save_link(url: str, status: str):
    """Insert a scanned link record into the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO blocked_links (url, status, timestamp)
        VALUES (?, ?, ?)
        """,
        (url, status, datetime.utcnow().isoformat()),
    )
    conn.commit()
    last_id = cursor.lastrowid
    conn.close()
    return last_id


# ─── Fetch All Records (optional helper) ─────────────────────────────────────
def get_all_media():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM scanned_media ORDER BY id DESC")
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_all_links():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM blocked_links ORDER BY id DESC")
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows
