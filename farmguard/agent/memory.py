"""Outcome memory in SQLite (AEGISFARM `memory.py` idea: success rate per species/zone/strategy),
extended with hour-of-night buckets and recency so the policy can reason about habituation."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
  event_id TEXT PRIMARY KEY, wall_time REAL, video_time REAL, species TEXT, label TEXT,
  zone TEXT, track_id INTEGER, vlm_species TEXT, vlm_behaviour TEXT, vlm_confidence TEXT,
  final_status TEXT, attempts INTEGER, escalated INTEGER, farmer_message TEXT, extra TEXT);
CREATE TABLE IF NOT EXISTS attempts (
  id INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT, attempt_no INTEGER, wall_time REAL,
  hour_bucket TEXT, species TEXT, sound TEXT, source TEXT, reason TEXT, status TEXT,
  success INTEGER, time_to_leave REAL, tracker_status TEXT, vlm_status TEXT);
CREATE INDEX IF NOT EXISTS idx_attempts_species ON attempts(species, sound);
"""


def hour_bucket(ts: float | None = None) -> str:
    h = time.localtime(ts or time.time()).tm_hour
    if 5 <= h < 11:
        return "morning"
    if 11 <= h < 17:
        return "day"
    if 17 <= h < 21:
        return "evening"
    return "night"


class Memory:
    def __init__(self, path: str | Path = ":memory:"):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._lock = threading.Lock()

    def add_event(self, **row: Any) -> None:
        extra = row.pop("extra", None)
        row["extra"] = json.dumps(extra, ensure_ascii=False) if extra is not None else None
        cols = ",".join(row)
        with self._lock:
            self.conn.execute(f"INSERT OR REPLACE INTO events ({cols}) VALUES "
                              f"({','.join('?' * len(row))})", list(row.values()))
            self.conn.commit()

    def add_attempt(self, event_id: str, attempt_no: int, species: str, sound: str, source: str,
                    reason: str, status: str, time_to_leave: float | None,
                    tracker_status: str | None = None, vlm_status: str | None = None,
                    wall_time: float | None = None, bucket: str | None = None) -> None:
        wall_time = wall_time or time.time()
        success = int(status in ("gone", "leaving"))
        with self._lock:
            self.conn.execute(
                "INSERT INTO attempts (event_id, attempt_no, wall_time, hour_bucket, species, sound,"
                " source, reason, status, success, time_to_leave, tracker_status, vlm_status)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (event_id, attempt_no, wall_time, bucket or hour_bucket(wall_time), species, sound,
                 source, reason, status, success, time_to_leave, tracker_status, vlm_status))
            self.conn.commit()

    def sound_stats(self, species: str, bucket: str | None = None,
                    recent_n: int = 5) -> dict[str, dict[str, Any]]:
        """Per sound: trials, successes, success_rate, recent outcomes (newest first)."""
        q = "SELECT sound, success, time_to_leave FROM attempts WHERE species=?"
        args: list[Any] = [species]
        if bucket:
            q += " AND hour_bucket=?"
            args.append(bucket)
        q += " ORDER BY id DESC"
        with self._lock:
            rows = self.conn.execute(q, args).fetchall()
        stats: dict[str, dict[str, Any]] = {}
        for r in rows:
            s = stats.setdefault(r["sound"], {"trials": 0, "successes": 0, "recent": []})
            s["trials"] += 1
            s["successes"] += r["success"]
            if len(s["recent"]) < recent_n:
                s["recent"].append("ok" if r["success"] else "fail")
        for s in stats.values():
            s["success_rate"] = round(s["successes"] / s["trials"], 2) if s["trials"] else None
        return stats

    def recent_events(self, since_ts: float) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute("SELECT * FROM events WHERE wall_time>=? ORDER BY wall_time",
                                     (since_ts,)).fetchall()
        return [dict(r) for r in rows]
