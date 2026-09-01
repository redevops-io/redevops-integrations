"""PydanticAI adapters for ReDevOps (frameworks plan Section 7).

Wraps a REAL `pydantic-ai` typed `Agent` as a ReDevOps Capability. PydanticAI keeps owning the typed agents,
typed dependencies/outputs, and the local pydantic-graph control flow; ReDevOps adds durable Mission identity,
context optimization, evidence, replay, authority, verification and governance around it. The Pydantic models stay
the local application types — canonical ReDevOps contracts are emitted only at the runtime boundary.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))

from adapter import NodeSpan  # noqa: E402

from pydantic import BaseModel  # noqa: E402
from pydantic_ai import Agent  # noqa: E402

try:
    import pydantic_ai as _pa
    PA_VERSION = getattr(_pa, "__version__", "2.36.0")
except Exception:
    PA_VERSION = "2.36.0"

os.environ.setdefault("CR_OPENAI_MODEL", "gpt-4o-mini")


class PydanticAgentCapability:
    """WorkflowAdapter over a real pydantic-ai typed Agent. Usable as a Mission capability handler (dict->dict).

    The typed `output_type` is the application's local Pydantic model; we project it to a JSON snapshot at the
    Mission boundary (never forcing runtime-contracts into the Pydantic node)."""

    framework = "pydantic-ai"
    framework_version = PA_VERSION

    def __init__(self, output_type: type[BaseModel], *, system_prompt: str,
                 model: str = "openai:gpt-4o-mini", name: str = "pydantic_agent"):
        self._agent = Agent(model, output_type=output_type, system_prompt=system_prompt)
        self._output_type = output_type
        self._name = name
        self.model_id = model
        self._last_span: NodeSpan | None = None
        self._last_output: BaseModel | None = None

    def invoke(self, capability: str, inputs: dict) -> dict:
        prompt = inputs.get("prompt") or inputs.get("question") or ""
        result = self._agent.run_sync(prompt)
        self._last_output = result.output
        self._last_span = NodeSpan(
            node_id=self.node_identity(capability, inputs), kind="agent", name=f"pydantic-ai:{self._name}",
            attrs={"framework": self.framework, "output_type": self._output_type.__name__},
            children=[NodeSpan(node_id="model", kind="model", name=self.model_id.split(":")[-1])])
        # boundary projection: the typed model → a JSON-able dict (Pydantic stays the local type)
        return {"output": result.output.model_dump(), "typed": True,
                "output_type": self._output_type.__name__, "framework": self.framework, "model": self.model_id}

    def node_identity(self, capability: str, inputs: dict) -> str:
        h = hashlib.sha256((capability + str(sorted(inputs.items()))).encode()).hexdigest()[:10]
        return f"pydantic-ai:{self._name}:{h}"

    def state_projection(self) -> dict:
        return {"framework": self.framework, "agent": self._name,
                "output_type": self._output_type.__name__,
                "output": self._last_output.model_dump() if self._last_output else None}

    def telemetry(self) -> NodeSpan:
        return self._last_span or NodeSpan(node_id=self._name, kind="agent", name=f"pydantic-ai:{self._name}")


class PydanticStateProjection:
    """Projects a typed Pydantic state object to a stable snapshot + content hash — used to prove that the local
    typed state AND the governed Mission state reconstruct identically under frozen replay (PA-C)."""

    @staticmethod
    def snapshot(state: BaseModel) -> dict:
        return state.model_dump(mode="json")

    @staticmethod
    def hash(state: BaseModel) -> str:
        return "sha256:" + hashlib.sha256(
            json.dumps(state.model_dump(mode="json"), sort_keys=True).encode()).hexdigest()[:16]
