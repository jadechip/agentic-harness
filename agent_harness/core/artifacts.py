from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _validate_schema_value(value: Any, schema: Any, path: str) -> list[str]:
    errors: list[str] = []

    if isinstance(schema, str):
        schema_type = schema.strip().lower()
        if schema_type.startswith("string"):
            if not isinstance(value, str):
                errors.append(f"{path}: expected string")
        elif schema_type.startswith("integer"):
            if not isinstance(value, int) or isinstance(value, bool):
                errors.append(f"{path}: expected integer")
        elif schema_type.startswith("float"):
            if not isinstance(value, (float, int)) or isinstance(value, bool):
                errors.append(f"{path}: expected float")
        elif schema_type.startswith("boolean"):
            if not isinstance(value, bool):
                errors.append(f"{path}: expected boolean")
        elif schema_type.startswith("object"):
            if not isinstance(value, dict):
                errors.append(f"{path}: expected object")
        elif schema_type.startswith("array"):
            if not isinstance(value, list):
                errors.append(f"{path}: expected array")
        return errors

    if isinstance(schema, list):
        if not isinstance(value, list):
            errors.append(f"{path}: expected list")
            return errors
        if not schema:
            return errors
        item_schema = schema[0]
        for idx, item in enumerate(value):
            errors.extend(_validate_schema_value(item, item_schema, f"{path}[{idx}]"))
        return errors

    if isinstance(schema, dict):
        if not isinstance(value, dict):
            errors.append(f"{path}: expected object")
            return errors

        if len(schema) == 1 and all(isinstance(k, str) and k.endswith("_path") for k in schema.keys()):
            dynamic_value_schema = list(schema.values())[0]
            for key, item_value in value.items():
                if not isinstance(key, str):
                    errors.append(f"{path}: expected string keys")
                    continue
                errors.extend(_validate_schema_value(item_value, dynamic_value_schema, f"{path}.{key}"))
            return errors

        for key, sub_schema in schema.items():
            if key not in value:
                errors.append(f"{path}.{key}: missing key")
                continue
            errors.extend(_validate_schema_value(value[key], sub_schema, f"{path}.{key}"))
        return errors

    return errors


@dataclass(frozen=True, slots=True)
class Artifact:
    id: str
    type: str
    schema_version: str
    content: dict[str, Any]
    produced_by_task: str
    created_at: datetime
    parent_artifact_id: str | None = None
    candidate_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "schema_version": self.schema_version,
            "content": self.content,
            "produced_by_task": self.produced_by_task,
            "created_at": self.created_at.isoformat(),
            "parent_artifact_id": self.parent_artifact_id,
            "candidate_index": self.candidate_index,
        }


class ArtifactFactory:
    """Factory to enforce artifact immutability and schema-valid creation semantics."""

    @staticmethod
    def create(
        artifact_type: str,
        schema_version: str,
        content: dict[str, Any],
        produced_by_task: str,
        schema: dict[str, Any] | None = None,
        parent_artifact_id: str | None = None,
        candidate_index: int | None = None,
    ) -> Artifact:
        if not isinstance(content, dict):
            raise ValueError("Artifact content must be a dictionary")

        if schema is not None:
            errors = _validate_schema_value(content, schema, artifact_type)
            if errors:
                raise ValueError("Artifact schema validation failed: " + "; ".join(errors[:10]))

        return Artifact(
            id=str(uuid4()),
            type=artifact_type,
            schema_version=schema_version,
            content=content,
            produced_by_task=produced_by_task,
            created_at=utc_now(),
            parent_artifact_id=parent_artifact_id,
            candidate_index=candidate_index,
        )
