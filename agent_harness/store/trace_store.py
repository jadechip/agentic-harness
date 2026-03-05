from __future__ import annotations

import json
import sqlite3
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
                    provider_name TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    temperature REAL NOT NULL DEFAULT 0,
                    max_tokens INTEGER NOT NULL DEFAULT 0,
                    evaluator_mode TEXT NOT NULL DEFAULT 'deterministic',
                    prompt TEXT NOT NULL,
                    tool_calls_json TEXT NOT NULL,
                    artifact_id TEXT NOT NULL,
                    candidate INTEGER NOT NULL,
                    selected INTEGER NOT NULL DEFAULT 0,
                    evaluation_score REAL NOT NULL,
                    evaluation_breakdown_json TEXT NOT NULL DEFAULT '{}',
                    token_usage INTEGER NOT NULL,
                    latency REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    passed INTEGER NOT NULL,
                    feedback_json TEXT NOT NULL
                )
                """
            )

            existing_columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(traces)").fetchall()
            }
            if "provider_name" not in existing_columns:
                conn.execute("ALTER TABLE traces ADD COLUMN provider_name TEXT NOT NULL DEFAULT ''")
            if "model_name" not in existing_columns:
                conn.execute("ALTER TABLE traces ADD COLUMN model_name TEXT NOT NULL DEFAULT ''")
            if "temperature" not in existing_columns:
                conn.execute("ALTER TABLE traces ADD COLUMN temperature REAL NOT NULL DEFAULT 0")
            if "max_tokens" not in existing_columns:
                conn.execute("ALTER TABLE traces ADD COLUMN max_tokens INTEGER NOT NULL DEFAULT 0")
            if "evaluator_mode" not in existing_columns:
                conn.execute(
                    "ALTER TABLE traces ADD COLUMN evaluator_mode TEXT NOT NULL DEFAULT 'deterministic'"
                )
            if "candidate" not in existing_columns:
                conn.execute("ALTER TABLE traces ADD COLUMN candidate INTEGER NOT NULL DEFAULT 1")
            if "selected" not in existing_columns:
                conn.execute("ALTER TABLE traces ADD COLUMN selected INTEGER NOT NULL DEFAULT 0")
            if "evaluation_breakdown_json" not in existing_columns:
                conn.execute(
                    "ALTER TABLE traces ADD COLUMN evaluation_breakdown_json TEXT NOT NULL DEFAULT '{}'"
                )

    def save(
        self,
        trace: Trace,
        attempt: int,
        passed: bool,
        feedback: list[str],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO traces (
                    run_id, task_name, skill_name, provider_name, model_name,
                    temperature, max_tokens, evaluator_mode,
                    prompt, tool_calls_json, artifact_id, candidate, selected,
                    evaluation_score, evaluation_breakdown_json, token_usage, latency,
                    created_at, attempt, passed, feedback_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace.run_id,
                    trace.task_name,
                    trace.skill_name,
                    trace.provider_name,
                    trace.model_name,
                    trace.temperature,
                    trace.max_tokens,
                    trace.evaluator_mode,
                    trace.prompt,
                    json.dumps(trace.tool_calls, sort_keys=True),
                    trace.artifact_id,
                    trace.candidate_index,
                    1 if trace.selected else 0,
                    trace.evaluation_score,
                    json.dumps(trace.evaluation_breakdown, sort_keys=True),
                    trace.token_usage,
                    trace.latency,
                    trace.created_at.isoformat(),
                    attempt,
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
                    "provider_name": row["provider_name"],
                    "model_name": row["model_name"],
                    "temperature": row["temperature"],
                    "max_tokens": row["max_tokens"],
                    "evaluator_mode": row["evaluator_mode"],
                    "prompt": row["prompt"],
                    "tool_calls": json.loads(row["tool_calls_json"]),
                    "artifact_id": row["artifact_id"],
                    "candidate": row["candidate"],
                    "selected": bool(row["selected"]),
                    "evaluation_score": row["evaluation_score"],
                    "evaluation_breakdown": json.loads(row["evaluation_breakdown_json"]),
                    "token_usage": row["token_usage"],
                    "latency": row["latency"],
                    "created_at": row["created_at"],
                    "attempt": row["attempt"],
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
