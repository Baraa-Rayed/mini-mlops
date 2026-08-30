"""Durable run state — the orchestrator's memory.

Maps to: Airflow's metadata database (`dag_run` + `task_instance` tables).

The single most important design point in the whole project: the scheduler
holds *no* state in memory. Kill it mid-run, start it again, and it resumes
from this database. That property is what makes an orchestrator an
orchestrator rather than a shell script with a for-loop.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path

# Task / run states. Deliberately the same vocabulary Airflow uses.
PENDING = "pending"
RUNNING = "running"
SUCCESS = "success"
FAILED = "failed"
UPSTREAM_FAILED = "upstream_failed"

TERMINAL = {SUCCESS, FAILED, UPSTREAM_FAILED}

HOME = Path(os.environ.get("MINI_HOME", Path.home() / ".mini-mlops"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    dag_id      TEXT NOT NULL,
    state       TEXT NOT NULL,
    trigger     TEXT NOT NULL,          -- 'manual' | 'schedule' | 'gitops'
    created_at  REAL NOT NULL,
    finished_at REAL,
    run_dir     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS task_instances (
    run_id      TEXT NOT NULL,
    task_id     TEXT NOT NULL,
    state       TEXT NOT NULL,
    try_number  INTEGER NOT NULL DEFAULT 0,
    started_at  REAL,
    finished_at REAL,
    error       TEXT,
    result      TEXT,                   -- JSON returned by the task (Airflow: XCom)
    PRIMARY KEY (run_id, task_id)
);
CREATE TABLE IF NOT EXISTS schedules (
    dag_id       TEXT PRIMARY KEY,
    last_run_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS applied (      -- GitOps: what commit we last synced
    app_name   TEXT PRIMARY KEY,
    revision   TEXT NOT NULL,
    synced_at  REAL NOT NULL
);
"""


class Store:
    def __init__(self, home: Path | str = HOME):
        self.home = Path(home)
        self.runs_dir = self.home / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        # The scheduler runs tasks on a thread pool, so the connection is
        # shared across threads. `check_same_thread=False` permits that;
        # `_lock` is what actually makes read-modify-write pairs (see
        # `bump_try`) safe. WAL keeps a reader from blocking on a writer,
        # which matters once a UI or a second process is watching a live run.
        self.db = sqlite3.connect(self.home / "mini.db", timeout=30, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.executescript(SCHEMA)
            self.db.commit()

    # --- runs ---------------------------------------------------------------
    def create_run(self, dag_id: str, trigger: str = "manual") -> sqlite3.Row:
        run_id = f"{dag_id}__{time.strftime('%Y%m%dT%H%M%S')}__{uuid.uuid4().hex[:6]}"
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self.db.execute(
                "INSERT INTO runs (run_id, dag_id, state, trigger, created_at, run_dir)"
                " VALUES (?,?,?,?,?,?)",
                (run_id, dag_id, RUNNING, trigger, time.time(), str(run_dir)),
            )
            self.db.commit()
            return self.get_run(run_id)

    def get_run(self, run_id: str) -> sqlite3.Row:
        return self.db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()

    def finish_run(self, run_id: str, state: str) -> None:
        with self._lock:
            self.db.execute(
                "UPDATE runs SET state=?, finished_at=? WHERE run_id=?", (state, time.time(), run_id)
            )
            self.db.commit()

    def list_runs(self, dag_id: str | None = None, limit: int = 20) -> list[sqlite3.Row]:
        sql = "SELECT * FROM runs"
        args: tuple = ()
        if dag_id:
            sql += " WHERE dag_id=?"
            args = (dag_id,)
        sql += " ORDER BY created_at DESC LIMIT ?"
        return self.db.execute(sql, args + (limit,)).fetchall()

    # --- task instances -----------------------------------------------------
    def init_task(self, run_id: str, task_id: str) -> None:
        with self._lock:
            self.db.execute(
                "INSERT OR IGNORE INTO task_instances (run_id, task_id, state) VALUES (?,?,?)",
                (run_id, task_id, PENDING),
            )
            self.db.commit()

    def set_task_state(self, run_id: str, task_id: str, state: str, **fields) -> None:
        cols = ", ".join(f"{k}=?" for k in fields)
        sql = f"UPDATE task_instances SET state=?{', ' + cols if cols else ''} WHERE run_id=? AND task_id=?"
        with self._lock:
            self.db.execute(sql, (state, *fields.values(), run_id, task_id))
            self.db.commit()

    def bump_try(self, run_id: str, task_id: str) -> int:
        # Increment-then-read: two statements that must not interleave with
        # another thread's attempt on the same task, hence the lock.
        with self._lock:
            self.db.execute(
                "UPDATE task_instances SET try_number = try_number + 1 WHERE run_id=? AND task_id=?",
                (run_id, task_id),
            )
            self.db.commit()
            row = self.db.execute(
                "SELECT try_number FROM task_instances WHERE run_id=? AND task_id=?", (run_id, task_id)
            ).fetchone()
        return row["try_number"]

    def task_states(self, run_id: str) -> dict[str, str]:
        rows = self.db.execute(
            "SELECT task_id, state FROM task_instances WHERE run_id=?", (run_id,)
        ).fetchall()
        return {r["task_id"]: r["state"] for r in rows}

    def task_states_full(self, run_id: str) -> dict[str, sqlite3.Row]:
        rows = self.db.execute(
            "SELECT * FROM task_instances WHERE run_id=?", (run_id,)
        ).fetchall()
        return {r["task_id"]: r for r in rows}

    def results(self, run_id: str) -> dict[str, dict]:
        """Every succeeded task's return value — this is our XCom."""
        rows = self.db.execute(
            "SELECT task_id, result FROM task_instances WHERE run_id=? AND state=?",
            (run_id, SUCCESS),
        ).fetchall()
        return {r["task_id"]: json.loads(r["result"] or "null") for r in rows}

    # --- scheduling ---------------------------------------------------------
    def last_run_at(self, dag_id: str) -> float | None:
        row = self.db.execute("SELECT last_run_at FROM schedules WHERE dag_id=?", (dag_id,)).fetchone()
        return row["last_run_at"] if row else None

    def mark_scheduled(self, dag_id: str, when: float) -> None:
        with self._lock:
            self.db.execute(
                "INSERT INTO schedules (dag_id, last_run_at) VALUES (?,?)"
                " ON CONFLICT(dag_id) DO UPDATE SET last_run_at=excluded.last_run_at",
                (dag_id, when),
            )
            self.db.commit()

    # --- gitops -------------------------------------------------------------
    def applied_revision(self, app_name: str) -> str | None:
        row = self.db.execute("SELECT revision FROM applied WHERE app_name=?", (app_name,)).fetchone()
        return row["revision"] if row else None

    def set_applied_revision(self, app_name: str, revision: str) -> None:
        with self._lock:
            self.db.execute(
                "INSERT INTO applied (app_name, revision, synced_at) VALUES (?,?,?)"
                " ON CONFLICT(app_name) DO UPDATE SET revision=excluded.revision, synced_at=excluded.synced_at",
                (app_name, revision, time.time()),
            )
            self.db.commit()
