"""The ReDevOps Workflow Adapter Contract (frameworks plan, Section 27).

A small, public seam so any agent framework can participate without bespoke Runtime changes. An adapter maps a
framework's agent/workflow onto the surfaces the Mission needs: invoke, node identity, state projection,
capability mapping, event/telemetry mapping, checkpoint/recovery, cancellation/containment. Implementing this
contract turns "support framework X" into a conformance exercise, not a new architecture project.

The NVIDIA adapter (`integrations/nvidia/adapters.py`) is the reference implementation; PydanticAI / LlamaIndex /
CrewAI / the hyperscaler managed stacks implement the same contract at their layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class NodeSpan:
    """One framework execution step, mapped for nesting UNDER the Mission causal trace (never above it)."""
    node_id: str
    kind: str                                # agent | tool | model | workflow_step | evaluator
    name: str
    attrs: dict[str, Any] = field(default_factory=dict)
    children: list["NodeSpan"] = field(default_factory=list)


@runtime_checkable
class WorkflowAdapter(Protocol):
    """What a framework must expose to run beneath a ReDevOps Mission. Deliberately minimal."""

    framework: str
    framework_version: str

    def invoke(self, capability: str, inputs: dict) -> dict:
        """Run one framework node/agent/tool. Pure dict->dict, so it drops straight into a Mission capability
        handler. MUST be idempotent given the same inputs + an idempotency key supplied by the Mission."""
        ...

    def node_identity(self, capability: str, inputs: dict) -> str:
        """Stable id for this step (framework task/agent id) — mapped to a Mission node."""

    def state_projection(self) -> dict:
        """Framework-local state projected to a JSON-able snapshot (for telemetry / diffing) — NOT the governed
        Runtime state, which the Mission owns."""

    def telemetry(self) -> NodeSpan:
        """The framework's execution tree for THIS invocation, to nest under the Mission node."""


def spans_to_dict(span: NodeSpan) -> dict:
    return {"node_id": span.node_id, "kind": span.kind, "name": span.name, "attrs": span.attrs,
            "children": [spans_to_dict(c) for c in span.children]}
