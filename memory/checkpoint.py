from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .sim_clock import SimulationClock
from .types import SimulationTime, jsonable


CHECKPOINT_SCHEMA_VERSION = 1
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class CheckpointError(RuntimeError):
    """A checkpoint is invalid, corrupted, or cannot be persisted."""


def _canonical(value: object) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise TypeError("checkpoint state must be JSON-compatible") from exc


def _detached(value: object) -> object:
    return json.loads(_canonical(jsonable(value)))


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed mapping")
    detached = _detached(dict(value))
    assert isinstance(detached, dict)
    return detached


def _records(value: object, name: str) -> tuple[dict[str, object], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of mappings")
    records = []
    for item in value:
        records.append(_mapping(item, f"{name} item"))
    return tuple(records)


@dataclass(frozen=True, slots=True, kw_only=True)
class SimulationCheckpoint:
    simulation_time: datetime
    portfolio_state: Mapping[str, object]
    shared_memory: Mapping[str, object]
    working_memory: Sequence[Mapping[str, object]]
    audit_offset: int
    pending_events: Sequence[Mapping[str, object]]
    schema_version: int = CHECKPOINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.simulation_time, datetime):
            raise TypeError("simulation_time must be datetime")
        if self.simulation_time.tzinfo is None:
            raise ValueError("simulation_time must be timezone-aware")
        if self.schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported checkpoint schema version: {self.schema_version}"
            )
        if isinstance(self.audit_offset, bool) or not isinstance(self.audit_offset, int):
            raise TypeError("audit_offset must be an integer")
        if self.audit_offset < 0:
            raise ValueError("audit_offset cannot be negative")
        object.__setattr__(
            self, "simulation_time", self.simulation_time.astimezone(timezone.utc)
        )
        object.__setattr__(
            self, "portfolio_state", _mapping(self.portfolio_state, "portfolio_state")
        )
        object.__setattr__(
            self, "shared_memory", _mapping(self.shared_memory, "shared_memory")
        )
        object.__setattr__(
            self, "working_memory", _records(self.working_memory, "working_memory")
        )
        object.__setattr__(
            self, "pending_events", _records(self.pending_events, "pending_events")
        )

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "simulation_time": self.simulation_time.isoformat(),
            "portfolio_state": deepcopy(dict(self.portfolio_state)),
            "shared_memory": deepcopy(dict(self.shared_memory)),
            "working_memory": deepcopy(list(self.working_memory)),
            "audit_offset": self.audit_offset,
            "pending_events": deepcopy(list(self.pending_events)),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> SimulationCheckpoint:
        expected = {
            "schema_version", "simulation_time", "portfolio_state", "shared_memory",
            "working_memory", "audit_offset", "pending_events",
        }
        if set(payload) != expected:
            missing = sorted(expected - set(payload))
            extra = sorted(set(payload) - expected)
            raise CheckpointError(
                f"checkpoint fields do not match schema; missing={missing}, extra={extra}"
            )
        raw_time = payload["simulation_time"]
        if not isinstance(raw_time, str):
            raise CheckpointError("simulation_time must be an ISO-8601 string")
        try:
            simulation_time = datetime.fromisoformat(raw_time)
        except ValueError as exc:
            raise CheckpointError("simulation_time is not valid ISO-8601") from exc
        try:
            return cls(
                schema_version=payload["schema_version"],
                simulation_time=simulation_time,
                portfolio_state=payload["portfolio_state"],
                shared_memory=payload["shared_memory"],
                working_memory=payload["working_memory"],
                audit_offset=payload["audit_offset"],
                pending_events=payload["pending_events"],
            )
        except (TypeError, ValueError) as exc:
            raise CheckpointError(f"invalid checkpoint payload: {exc}") from exc

    def resume(self) -> SimulationResumeState:
        return SimulationResumeState(
            clock=SimulationClock(SimulationTime(self.simulation_time)),
            portfolio_state=deepcopy(dict(self.portfolio_state)),
            shared_memory=deepcopy(dict(self.shared_memory)),
            working_memory=deepcopy(tuple(self.working_memory)),
            audit_offset=self.audit_offset,
            pending_events=deepcopy(tuple(self.pending_events)),
        )


@dataclass(frozen=True, slots=True)
class SimulationResumeState:
    clock: SimulationClock
    portfolio_state: Mapping[str, object]
    shared_memory: Mapping[str, object]
    working_memory: tuple[Mapping[str, object], ...]
    audit_offset: int
    pending_events: tuple[Mapping[str, object], ...]


class CheckpointStore:
    """Atomic, checksummed checkpoint persistence inside one directory."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory).resolve()

    def save(self, name: str, checkpoint: SimulationCheckpoint) -> Path:
        path = self._path(name)
        if not isinstance(checkpoint, SimulationCheckpoint):
            raise TypeError("checkpoint must be SimulationCheckpoint")
        payload = checkpoint.payload()
        envelope = {
            "checksum": hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest(),
            "payload": payload,
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            temporary.write_text(_canonical(envelope), encoding="utf-8")
            temporary.replace(path)
        except OSError as exc:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise CheckpointError(f"cannot persist checkpoint: {path}") from exc
        return path

    def load(self, name: str) -> SimulationCheckpoint:
        path = self._path(name)
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointError(f"cannot load checkpoint: {path}") from exc
        if not isinstance(envelope, dict) or set(envelope) != {"checksum", "payload"}:
            raise CheckpointError("checkpoint envelope has an invalid format")
        checksum = envelope["checksum"]
        payload = envelope["payload"]
        if not isinstance(checksum, str) or not isinstance(payload, dict):
            raise CheckpointError("checkpoint envelope has invalid values")
        actual = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
        if actual != checksum:
            raise CheckpointError("checkpoint checksum mismatch")
        return SimulationCheckpoint.from_payload(payload)

    def _path(self, name: str) -> Path:
        if not isinstance(name, str) or _NAME.fullmatch(name) is None:
            raise ValueError("checkpoint name contains unsupported characters")
        return self.directory / f"{name}.json"
