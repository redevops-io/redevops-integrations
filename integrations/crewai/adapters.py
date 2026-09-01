"""CrewAI adapters for ReDevOps (frameworks plan Section 7).

Wraps REAL `crewai` Agents / Crews / Tools. CrewAI keeps owning role agents, crews, tasks, the delegation tool and
the hierarchical/sequential process; ReDevOps adds a durable Mission around the crew and — the CrewAI-specific
boundary — an **authority envelope that narrows across delegation**. In CrewAI, when an agent delegates to a
coworker the coworker runs with its own tools; there is no notion that a delegate may not exceed the delegator's
authority. ReDevOps makes a delegated agent's effective grants the *intersection* of the chain (monotone
non-increasing), and admits a consequential tool call only if its required permission is inside that envelope.
"""
from __future__ import annotations

import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))

from adapter import NodeSpan  # noqa: E402

from crewai.tools import BaseTool  # noqa: E402

try:
    import crewai as _cw
    CREWAI_VERSION = getattr(_cw, "__version__", "1.15.18")
except Exception:
    CREWAI_VERSION = "1.15.18"

os.environ.setdefault("OPENAI_MODEL_NAME", "gpt-4o-mini")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")


class AuthorityEnvelope:
    """The grants an agent may act under. Delegation NARROWS it — a child's effective envelope is the intersection
    of its declared grants with its delegator's effective envelope. Authority never widens across a handoff."""

    def __init__(self, grants, *, agent="", parent: "AuthorityEnvelope | None" = None):
        declared = frozenset(grants)
        self.grants = declared if parent is None else (declared & parent.grants)
        self.agent = agent
        self.parent = parent

    def narrow(self, child_grants, *, agent="") -> "AuthorityEnvelope":
        return AuthorityEnvelope(child_grants, agent=agent, parent=self)

    def admits(self, permission: str) -> bool:
        return permission in self.grants

    def chain(self) -> list[str]:
        return ([] if self.parent is None else self.parent.chain()) + [self.agent or "?"]


class RedevopsGovernedTool(BaseTool):
    """A REAL crewai tool whose consequential action is admitted only when the acting agent's (possibly delegated,
    hence narrowed) authority envelope permits it. Side effects are recorded so exactly-once / governance can be
    asserted. Native CrewAI would run the tool whenever the agent holds it — this adds the authority gate."""

    name: str = "governed_action"
    description: str = "Perform a consequential business action, subject to the agent's authority envelope."
    permission: str = ""

    # non-pydantic runtime state
    _envelope: object = None
    _sink: object = None

    def bind(self, envelope: AuthorityEnvelope, sink: list):
        object.__setattr__(self, "_envelope", envelope)
        object.__setattr__(self, "_sink", sink)
        return self

    def _run(self, *args, **kwargs) -> str:
        env: AuthorityEnvelope = self._envelope
        if env is None or not env.admits(self.permission):
            return f"DENIED: '{self.permission}' exceeds the delegated authority of {env.agent if env else '?'}"
        self._sink.append(self.permission)
        return f"OK: performed '{self.permission}'"


class CrewCapability:
    """WorkflowAdapter over a real crewai Crew — usable as a Mission capability handler and telemetry source."""

    framework = "crewai"
    framework_version = CREWAI_VERSION

    def __init__(self, crew, *, name: str = "crew"):
        self._crew = crew
        self._name = name
        self.model_id = os.environ.get("OPENAI_MODEL_NAME", "gpt-4o-mini")
        self._last_span: NodeSpan | None = None

    def invoke(self, capability: str, inputs: dict) -> dict:
        out = self._crew.kickoff(inputs=inputs.get("inputs") or {})
        self._last_span = self.telemetry()
        return {"output": str(out).strip(), "framework": self.framework, "model": self.model_id}

    def node_identity(self, capability: str, inputs: dict) -> str:
        h = hashlib.sha256((capability + str(sorted(inputs.items()))).encode()).hexdigest()[:10]
        return f"crewai:{self._name}:{h}"

    def state_projection(self) -> dict:
        return {"framework": self.framework, "crew": self._name,
                "agents": [getattr(a, "role", "?") for a in getattr(self._crew, "agents", [])]}

    def telemetry(self) -> NodeSpan:
        agents = getattr(self._crew, "agents", [])
        kids = [NodeSpan(node_id=f"agent:{i}", kind="agent", name=getattr(a, "role", "agent"),
                         children=[NodeSpan(node_id="model", kind="model", name=self.model_id)])
                for i, a in enumerate(agents)]
        return NodeSpan(node_id=f"crewai:{self._name}", kind="crew", name=f"crewai:{self._name}", children=kids)
