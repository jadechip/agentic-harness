from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class Artifact:
    id: str
    type: str
    schema_version: str
    content: dict[str, Any]
    produced_by_task: str
    created_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "schema_version": self.schema_version,
            "content": self.content,
            "produced_by_task": self.produced_by_task,
            "created_at": self.created_at.isoformat(),
        }


class ArtifactFactory:
    """Factory to enforce artifact immutability and stable creation semantics."""

    @staticmethod
    def create(
        artifact_type: str,
        schema_version: str,
        content: dict[str, Any],
        produced_by_task: str,
    ) -> Artifact:
        return Artifact(
            id=str(uuid4()),
            type=artifact_type,
            schema_version=schema_version,
            content=content,
            produced_by_task=produced_by_task,
            created_at=utc_now(),
        )
