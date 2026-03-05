from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._ensure_table()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_table(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    harness_name TEXT NOT NULL,
                    user_request TEXT NOT NULL,
                    repo_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    metadata_json TEXT NOT NULL
                )
                """
            )

    def start_run(
        self,
        harness_name: str,
        user_request: str,
        repo_path: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        run_id = str(uuid4())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs (
                    run_id, harness_name, user_request, repo_path,
                    status, started_at, completed_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    harness_name,
                    user_request,
                    repo_path,
                    "running",
                    _utc_iso(),
                    None,
                    json.dumps(metadata or {}, sort_keys=True),
                ),
            )
        return run_id

    def finish_run(self, run_id: str, status: str, metadata: dict[str, Any] | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE runs
                SET status = ?, completed_at = ?, metadata_json = ?
                WHERE run_id = ?
                """,
                (status, _utc_iso(), json.dumps(metadata or {}, sort_keys=True), run_id),
            )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()

        if row is None:
            return None
        return {
            "run_id": row["run_id"],
            "harness_name": row["harness_name"],
            "user_request": row["user_request"],
            "repo_path": row["repo_path"],
            "status": row["status"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
            "metadata": json.loads(row["metadata_json"]),
        }
