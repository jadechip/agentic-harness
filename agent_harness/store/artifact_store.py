from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from agent_harness.core.artifacts import Artifact


class ArtifactStore:
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
                CREATE TABLE IF NOT EXISTS artifacts (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    task_name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    content_json TEXT NOT NULL,
                    produced_by_task TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    parent_artifact_id TEXT,
                    candidate_index INTEGER,
                    attempt INTEGER NOT NULL,
                    candidate INTEGER NOT NULL,
                    selected INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS selected_artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    task_name TEXT NOT NULL,
                    artifact_id TEXT NOT NULL,
                    selected_at TEXT NOT NULL
                )
                """
            )

            existing_columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(artifacts)").fetchall()
            }
            if "parent_artifact_id" not in existing_columns:
                conn.execute("ALTER TABLE artifacts ADD COLUMN parent_artifact_id TEXT")
            if "candidate_index" not in existing_columns:
                conn.execute("ALTER TABLE artifacts ADD COLUMN candidate_index INTEGER")

    def save(
        self,
        run_id: str,
        task_name: str,
        artifact: Artifact,
        attempt: int,
        candidate: int,
        selected: bool,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO artifacts (
                    id, run_id, task_name, type, schema_version, content_json,
                    produced_by_task, created_at, parent_artifact_id, candidate_index,
                    attempt, candidate, selected
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact.id,
                    run_id,
                    task_name,
                    artifact.type,
                    artifact.schema_version,
                    json.dumps(artifact.content, sort_keys=True),
                    artifact.produced_by_task,
                    artifact.created_at.isoformat(),
                    artifact.parent_artifact_id,
                    artifact.candidate_index,
                    attempt,
                    candidate,
                    1 if selected else 0,
                ),
            )

    def mark_selected(self, run_id: str, task_name: str, artifact_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO selected_artifacts (run_id, task_name, artifact_id, selected_at)
                VALUES (?, ?, ?, datetime('now'))
                """,
                (run_id, task_name, artifact_id),
            )

    def get_latest_selected(self, run_id: str, artifact_type: str) -> Artifact | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT a.*
                FROM artifacts a
                JOIN (
                    SELECT sa.task_name, sa.artifact_id
                    FROM selected_artifacts sa
                    JOIN (
                        SELECT task_name, MAX(id) AS max_id
                        FROM selected_artifacts
                        WHERE run_id = ?
                        GROUP BY task_name
                    ) latest ON latest.task_name = sa.task_name AND latest.max_id = sa.id
                    WHERE sa.run_id = ?
                ) selected ON selected.artifact_id = a.id
                WHERE a.type = ?
                ORDER BY a.created_at DESC
                LIMIT 1
                """,
                (run_id, run_id, artifact_type),
            ).fetchone()

        if row is None:
            return None
        return self._row_to_artifact(row)

    def list_selected_for_run(self, run_id: str) -> list[Artifact]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT a.*
                FROM artifacts a
                JOIN (
                    SELECT sa.task_name, sa.artifact_id
                    FROM selected_artifacts sa
                    JOIN (
                        SELECT task_name, MAX(id) AS max_id
                        FROM selected_artifacts
                        WHERE run_id = ?
                        GROUP BY task_name
                    ) latest ON latest.task_name = sa.task_name AND latest.max_id = sa.id
                    WHERE sa.run_id = ?
                ) selected ON selected.artifact_id = a.id
                ORDER BY a.created_at ASC
                """,
                (run_id, run_id),
            ).fetchall()
        return [self._row_to_artifact(row) for row in rows]

    @staticmethod
    def _row_to_artifact(row: sqlite3.Row) -> Artifact:
        return Artifact(
            id=row["id"],
            type=row["type"],
            schema_version=row["schema_version"],
            content=json.loads(row["content_json"]),
            produced_by_task=row["produced_by_task"],
            created_at=datetime.fromisoformat(row["created_at"]),
            parent_artifact_id=row["parent_artifact_id"],
            candidate_index=row["candidate_index"],
        )
