from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reasoning_agent_template.models import utc_now


_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


@dataclass(frozen=True)
class SessionSnapshot:
    session_id: str
    status: str
    messages: list[dict[str, Any]]
    events: list[dict[str, Any]]
    updated_at: str
    parent_session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "session_id": self.session_id,
            "parent_session_id": self.parent_session_id,
            "status": self.status,
            "updated_at": self.updated_at,
            "messages": self.messages,
            "events": self.events,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionSnapshot":
        return cls(
            session_id=str(data["session_id"]),
            parent_session_id=(
                str(data["parent_session_id"])
                if data.get("parent_session_id") is not None
                else None
            ),
            status=str(data.get("status", "unknown")),
            updated_at=str(data.get("updated_at", "")),
            messages=list(data.get("messages", [])),
            events=list(data.get("events", [])),
            metadata=dict(data.get("metadata", {})),
        )


class SessionStore:
    """Local transcript store used as the base for resume/fork/background work."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def record_snapshot(
        self,
        *,
        session_id: str,
        messages: list[dict[str, Any]],
        events: list[dict[str, Any]],
        status: str,
        parent_session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SessionSnapshot:
        session_id = self._validate_session_id(session_id)
        if parent_session_id is not None:
            parent_session_id = self._validate_session_id(parent_session_id)
        snapshot = SessionSnapshot(
            session_id=session_id,
            parent_session_id=parent_session_id,
            status=status,
            updated_at=utc_now(),
            messages=[dict(message) for message in messages],
            events=[dict(event) for event in events],
            metadata=dict(metadata or {}),
        )
        self._write(snapshot)
        return snapshot

    def load(self, session_id: str) -> SessionSnapshot:
        session_id = self._validate_session_id(session_id)
        path = self._snapshot_path(session_id)
        data = json.loads(path.read_text(encoding="utf-8"))
        return SessionSnapshot.from_dict(data)

    def load_messages(self, session_id: str) -> list[dict[str, Any]]:
        return self.load(session_id).messages

    def fork(self, source_session_id: str, new_session_id: str) -> SessionSnapshot:
        source = self.load(source_session_id)
        event = {
            "time": utc_now(),
            "kind": "session_forked",
            "source_session_id": source.session_id,
            "session_id": new_session_id,
        }
        return self.record_snapshot(
            session_id=new_session_id,
            parent_session_id=source.session_id,
            messages=source.messages,
            events=[*source.events, event],
            status=source.status,
            metadata={**source.metadata, "forked_from": source.session_id},
        )

    def delete(self, session_id: str) -> None:
        path = self._snapshot_path(session_id)
        if path.exists():
            path.unlink()
        parent = path.parent
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()

    def _write(self, snapshot: SessionSnapshot) -> None:
        path = self._snapshot_path(snapshot.session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(snapshot.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _snapshot_path(self, session_id: str) -> Path:
        session_id = self._validate_session_id(session_id)
        root = self.root.resolve()
        path = (root / session_id / "snapshot.json").resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"session id escapes session store: {session_id}") from exc
        return path

    @staticmethod
    def _validate_session_id(session_id: str) -> str:
        value = str(session_id).strip()
        if not _SESSION_ID_RE.fullmatch(value):
            raise ValueError(
                "session_id must start with an alphanumeric character and may only contain letters, numbers, dots, underscores, and hyphens"
            )
        return value
