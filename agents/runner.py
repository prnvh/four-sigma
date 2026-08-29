from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from time import perf_counter
from typing import Any, Protocol

from memory.audit_logger import AuditEventType, AuditLedger
from memory.capabilities import Action, CAPABILITIES, CapabilityModel
from memory.context_gateway import ContextGateway, ContextPurpose
from memory.types import (
    AgentId,
    AuditEventId,
    ContextSnapshotId,
    CreatedAt,
    EntityId,
    OutcomeDefinition,
    RunId,
    jsonable,
)
from memory.working_mem import WorkingMemory, WorkingMemoryCategory, WorkingMemoryEntry

from .contracts import StructuredAgentOutput
from .registry import AgentSpec


class RunnerModel(Protocol):
    def generate_json(
        self, *, instructions: str, input_data: dict[str, Any], schema: dict[str, Any]
    ) -> object: ...


@dataclass(frozen=True, slots=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int

    def __post_init__(self) -> None:
        for value in (self.input_tokens, self.output_tokens):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("token counts must be non-negative integers")

    def as_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
        }


@dataclass(frozen=True, slots=True)
class ModelCallResult:
    output: object
    token_usage: TokenUsage | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentRunRequest:
    run_id: RunId
    entity_id: EntityId
    purpose: ContextPurpose
    simulation_time: datetime
    working_category: WorkingMemoryCategory = WorkingMemoryCategory.CANDIDATE_INSIGHT
    requested_fields: tuple[tuple[str, str], ...] = ()
    outcome: OutcomeDefinition | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, RunId):
            raise TypeError("run_id must be RunId")
        if not isinstance(self.entity_id, EntityId):
            raise TypeError("entity_id must be EntityId")
        if not isinstance(self.purpose, ContextPurpose):
            raise TypeError("purpose must be ContextPurpose")
        if not isinstance(self.simulation_time, datetime):
            raise TypeError("simulation_time must be datetime")
        if self.simulation_time.tzinfo is None:
            raise ValueError("simulation_time must be timezone-aware")
        if not isinstance(self.working_category, WorkingMemoryCategory):
            raise TypeError("working_category must be WorkingMemoryCategory")


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    output: StructuredAgentOutput
    working_memory_entry: WorkingMemoryEntry
    context_snapshot_id: ContextSnapshotId
    latency_ms: float
    token_usage: TokenUsage | None


class AgentRunner:
    """One permissioned, validated and audited execution path for agents."""

    def __init__(
        self,
        *,
        context_gateway: ContextGateway,
        working_memory: WorkingMemory,
        audit_ledger: AuditLedger,
        capabilities: CapabilityModel | None = None,
    ) -> None:
        self.context_gateway = context_gateway
        self.working_memory = working_memory
        self.audit_ledger = audit_ledger
        self.capabilities = capabilities or CAPABILITIES

    def run(
        self,
        request: AgentRunRequest,
        *,
        spec: AgentSpec,
        model: RunnerModel,
        model_name: str,
        output_type: type[StructuredAgentOutput],
    ) -> AgentRunResult:
        if not isinstance(request, AgentRunRequest):
            raise TypeError("request must be AgentRunRequest")
        if not isinstance(spec, AgentSpec):
            raise TypeError("spec must be AgentSpec")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be non-empty")
        if not isinstance(output_type, type) or not issubclass(
            output_type, StructuredAgentOutput
        ):
            raise TypeError("output_type must be a structured agent output")
        expected_output = spec.config.get("output")
        if expected_output is not None and expected_output != output_type.contract_name:
            raise ValueError(
                f"{spec.key} requires the {expected_output!r} output contract"
            )

        agent_id = AgentId(spec.name)
        occurred_at = CreatedAt(request.simulation_time)
        snapshot_id: ContextSnapshotId | None = None
        usage: TokenUsage | None = None
        started = perf_counter()
        identity = {
            "agent_version": spec.version,
            "model": model_name.strip(),
            "prompt_version": spec.version,
        }
        self._audit(
            request, "started", AuditEventType.AGENT_RUN_STARTED,
            occurred_at, identity, agent_id,
        )
        try:
            self.capabilities.require(
                agent_id,
                Action.WORKING_WRITE,
                "working",
                request.working_category.value,
            )
            self._audit(
                request, "context-requested", AuditEventType.CONTEXT_REQUESTED,
                occurred_at,
                {"purpose": request.purpose.value, "entity_id": request.entity_id.value},
                agent_id,
            )
            bundle = self.context_gateway.get_context(
                agent_id=spec.name,
                purpose=request.purpose,
                entity_ids=(request.entity_id.value,),
                simulation_time=request.simulation_time,
                fields=request.requested_fields,
                outcome=request.outcome,
            )
            snapshot_id = bundle.snapshot.id
            self._audit(
                request, "context-returned", AuditEventType.CONTEXT_RETURNED,
                occurred_at, {"context_snapshot_id": snapshot_id.value}, agent_id,
            )
            raw = model.generate_json(
                instructions=spec.prompt,
                input_data=jsonable(bundle.view),
                schema=dict(output_type.schema),
            )
            if isinstance(raw, ModelCallResult):
                usage = raw.token_usage
                raw = raw.output
            output = output_type.parse(raw)
            entry = self.working_memory.for_agent(agent_id).write(
                audit_event_id=self._event_id(request, "working-memory"),
                run_id=request.run_id,
                entity_id=request.entity_id,
                category=request.working_category,
                value=output,
                created_at=occurred_at,
            )
        except Exception as error:
            self._finish(
                request, occurred_at, identity, agent_id, started,
                snapshot_id, usage, status="failed", error=error,
            )
            raise

        latency_ms = self._finish(
            request, occurred_at, identity, agent_id, started,
            snapshot_id, usage, status="succeeded",
        )
        return AgentRunResult(output, entry, snapshot_id, latency_ms, usage)

    def _finish(
        self,
        request: AgentRunRequest,
        occurred_at: CreatedAt,
        identity: Mapping[str, object],
        agent_id: AgentId,
        started: float,
        snapshot_id: ContextSnapshotId | None,
        usage: TokenUsage | None,
        *,
        status: str,
        error: Exception | None = None,
    ) -> float:
        latency_ms = (perf_counter() - started) * 1000
        self._audit(
            request,
            "finished",
            AuditEventType.AGENT_RUN_FINISHED,
            occurred_at,
            {
                **identity,
                "status": status,
                "latency_ms": latency_ms,
                "token_usage": usage.as_dict() if usage else None,
                "context_snapshot_id": snapshot_id.value if snapshot_id else None,
                "error_type": type(error).__name__ if error else None,
            },
            agent_id,
        )
        return latency_ms

    def _audit(
        self,
        request: AgentRunRequest,
        suffix: str,
        event_type: AuditEventType,
        occurred_at: CreatedAt,
        details: Mapping[str, object],
        agent_id: AgentId,
    ) -> None:
        self.audit_ledger.append(
            event_id=self._event_id(request, suffix),
            event_type=event_type,
            occurred_at=occurred_at,
            details=details,
            agent_id=agent_id,
            run_id=request.run_id,
            subject_id=request.entity_id,
        )

    @staticmethod
    def _event_id(request: AgentRunRequest, suffix: str) -> AuditEventId:
        return AuditEventId(f"{request.run_id.value}:{suffix}")
