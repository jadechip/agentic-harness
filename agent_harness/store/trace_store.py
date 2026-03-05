from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from agent_harness.core.traces import Trace


class TraceStore:
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
                CREATE TABLE IF NOT EXISTS traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    task_name TEXT NOT NULL,
                    skill_name TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    tool_calls_json TEXT NOT NULL,
                    artifact_id TEXT NOT NULL,
                    evaluation_score REAL NOT NULL,
                    token_usage INTEGER NOT NULL,
                    latency REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    candidate INTEGER NOT NULL,
                    passed INTEGER NOT NULL,
                    feedback_json TEXT NOT NULL
                )
                """
            )

    def save(
        self,
        trace: Trace,
        attempt: int,
        candidate: int,
        passed: bool,
        feedback: list[str],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO traces (
                    run_id, task_name, skill_name, prompt, tool_calls_json,
                    artifact_id, evaluation_score, token_usage, latency, created_at,
                    attempt, candidate, passed, feedback_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace.run_id,
                    trace.task_name,
                    trace.skill_name,
                    trace.prompt,
                    json.dumps(trace.tool_calls, sort_keys=True),
                    trace.artifact_id,
                    trace.evaluation_score,
                    trace.token_usage,
                    trace.latency,
                    trace.created_at.isoformat(),
                    attempt,
                    candidate,
                    1 if passed else 0,
                    json.dumps(feedback),
                ),
            )

    def export_run(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM traces WHERE run_id = ? ORDER BY id ASC",
                (run_id,),
            ).fetchall()

        exported: list[dict[str, Any]] = []
        for row in rows:
            exported.append(
                {
                    "run_id": row["run_id"],
                    "task_name": row["task_name"],
                    "skill_name": row["skill_name"],
                    "prompt": row["prompt"],
                    "tool_calls": json.loads(row["tool_calls_json"]),
                    "artifact_id": row["artifact_id"],
                    "evaluation_score": row["evaluation_score"],
                    "token_usage": row["token_usage"],
                    "latency": row["latency"],
                    "created_at": row["created_at"],
                    "attempt": row["attempt"],
                    "candidate": row["candidate"],
                    "passed": bool(row["passed"]),
                    "feedback": json.loads(row["feedback_json"]),
                }
            )
        return exported

    def list_run_scores(self, run_id: str) -> list[float]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT evaluation_score FROM traces WHERE run_id = ?",
                (run_id,),
            ).fetchall()
        return [float(row["evaluation_score"]) for row in rows]
