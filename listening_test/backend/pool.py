"""Pool of audio samples to describe, organised in rounds and stored in SQLite.

Within a round, each sample is drawn without replacement.
"""

import glob
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

CLAIM_STALE_SECONDS = 30 * 60


class Pool:
    def __init__(self, db_path: str, audio_dir: str, allowed_ids: Optional[set] = None):
        self.db_path = db_path
        self.audio_dir = audio_dir
        self.allowed_ids = allowed_ids
        self._init_db()
        self._sync_samples()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self):
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS samples (
                    sample_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL DEFAULT 'pending',
                    claimed_by TEXT,
                    claimed_at REAL,
                    times_described INTEGER NOT NULL DEFAULT 0,
                    last_described_at REAL,
                    last_completed_round INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            cols = {row[1] for row in conn.execute("PRAGMA table_info(samples)")}
            if "last_completed_round" not in cols:
                conn.execute(
                    "ALTER TABLE samples ADD COLUMN last_completed_round "
                    "INTEGER NOT NULL DEFAULT 0"
                )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS round_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    current_round INTEGER NOT NULL DEFAULT 1
                )
                """
            )
            conn.execute("INSERT OR IGNORE INTO round_state (id, current_round) VALUES (1, 1)")

    def _sync_samples(self):
        wavs = sorted(glob.glob(os.path.join(self.audio_dir, "*.wav")))
        ids = [Path(w).stem for w in wavs]
        if self.allowed_ids is not None:
            ids = [i for i in ids if i in self.allowed_ids]
        if not ids:
            return
        with self._conn() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO samples (sample_id, status) VALUES (?, 'pending')",
                [(i,) for i in ids],
            )
            if self.allowed_ids is not None:
                conn.execute(
                    f"DELETE FROM samples WHERE sample_id NOT IN "
                    f"({','.join('?' for _ in ids)})",
                    ids,
                )

    def _current_round(self, conn) -> int:
        return conn.execute("SELECT current_round FROM round_state WHERE id=1").fetchone()[0]

    def _pick_eligible(self, conn, current_round: int, exclude: Optional[str]):
        if exclude:
            row = conn.execute(
                "SELECT sample_id FROM samples WHERE status='pending' "
                "AND last_completed_round < ? AND sample_id != ? "
                "ORDER BY RANDOM() LIMIT 1",
                (current_round, exclude),
            ).fetchone()
            if row:
                return row[0]
        row = conn.execute(
            "SELECT sample_id FROM samples WHERE status='pending' "
            "AND last_completed_round < ? ORDER BY RANDOM() LIMIT 1",
            (current_round,),
        ).fetchone()
        return row[0] if row else None

    def claim_next(self, session_id: str, exclude: Optional[str] = None) -> Optional[str]:
        now = time.time()
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT sample_id FROM samples WHERE status='in_progress' AND claimed_by=?",
                (session_id,),
            ).fetchone()
            if row:
                return row[0]

            current_round = self._current_round(conn)

            sample_id = self._pick_eligible(conn, current_round, exclude)

            if sample_id is None:
                stale_before = now - CLAIM_STALE_SECONDS
                row = conn.execute(
                    "SELECT sample_id FROM samples WHERE status='in_progress' "
                    "AND last_completed_round < ? AND claimed_at < ? "
                    "ORDER BY RANDOM() LIMIT 1",
                    (current_round, stale_before),
                ).fetchone()
                if row:
                    sample_id = row[0]

            if sample_id is None:
                (still_in_progress,) = conn.execute(
                    "SELECT COUNT(*) FROM samples WHERE status='in_progress' "
                    "AND last_completed_round < ?",
                    (current_round,),
                ).fetchone()
                if still_in_progress > 0:
                    return None

                conn.execute(
                    "UPDATE round_state SET current_round = current_round + 1 WHERE id=1"
                )
                current_round += 1
                sample_id = self._pick_eligible(conn, current_round, exclude)
                if sample_id is None:
                    return None

            conn.execute(
                "UPDATE samples SET status='in_progress', claimed_by=?, claimed_at=? "
                "WHERE sample_id=?",
                (session_id, now, sample_id),
            )
            return sample_id

    def record_description(self, session_id: str, sample_id: str) -> Optional[int]:
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT claimed_by, times_described FROM samples "
                "WHERE sample_id=? AND status='in_progress'",
                (sample_id,),
            ).fetchone()
            if not row or row[0] != session_id:
                return None
            new_count = row[1] + 1
            current_round = self._current_round(conn)
            conn.execute(
                "UPDATE samples SET status='pending', claimed_by=NULL, claimed_at=NULL, "
                "times_described=?, last_completed_round=?, last_described_at=? "
                "WHERE sample_id=?",
                (new_count, current_round, time.time(), sample_id),
            )
            return new_count

    def skip(self, session_id: str, sample_id: str) -> bool:
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT claimed_by FROM samples WHERE sample_id=? AND status='in_progress'",
                (sample_id,),
            ).fetchone()
            if not row or row[0] != session_id:
                return False
            conn.execute(
                "UPDATE samples SET status='pending', claimed_by=NULL, claimed_at=NULL "
                "WHERE sample_id=?",
                (sample_id,),
            )
            return True

    def stats(self) -> dict:
        with self._conn() as conn:
            current_round = self._current_round(conn)
            (total,) = conn.execute("SELECT COUNT(*) FROM samples").fetchone()
            (completed_this_round,) = conn.execute(
                "SELECT COUNT(*) FROM samples WHERE last_completed_round = ?",
                (current_round,),
            ).fetchone()
            (in_progress,) = conn.execute(
                "SELECT COUNT(*) FROM samples WHERE status='in_progress'"
            ).fetchone()
            (descriptions_collected,) = conn.execute(
                "SELECT COALESCE(SUM(times_described), 0) FROM samples"
            ).fetchone()
        return {
            "total": total,
            "current_round": current_round,
            "completed_this_round": completed_this_round,
            "pending_this_round": total - completed_this_round - in_progress,
            "in_progress": in_progress,
            "descriptions_collected": descriptions_collected,
        }
