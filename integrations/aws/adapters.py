"""AWS (Strands Agents) adapters for ReDevOps (frameworks plan Section 7 / hyperscaler plan).

Wraps a REAL AWS Strands Agents `Agent` (model-provider agnostic; pointed at any OpenAI-compatible endpoint) as a
ReDevOps capability. Strands / Bedrock AgentCore keep owning agent construction and the agent loop; ReDevOps adds
Mission semantics around it and — the AWS-specific boundary — authority that COMPOSES with IAM using AWS's own
evaluation rule: an explicit Deny overrides any Allow (deny-wins), and a consequential action is admitted only when
IAM allows it, no guardrail (SCP) denies it, and the Mission grants it.

Runs entirely locally (Strands + an OpenAI-compatible model) — no AWS credentials. IAM policy is modelled as a
policy document evaluated with AWS semantics, not a live cloud call. Coexists in the shared benchmark venv.
"""
from __future__ import annotations

import fnmatch
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))

from adapter import NodeSpan  # noqa: E402

try:
    import strands as _strands
    STRANDS_VERSION = getattr(_strands, "__version__", "1.54.0")
except Exception:
    STRANDS_VERSION = "1.54.0"


# ---- IAM-semantic policy evaluation (explicit Deny overrides Allow) ----
class Statement:
    def __init__(self, effect: str, actions, resources=("*",)):
        self.effect = effect                 # "Allow" | "Deny"
        self.actions = list(actions)
        self.resources = list(resources)

    def matches(self, action: str, resource: str) -> bool:
        a = any(fnmatch.fnmatch(action, pat) for pat in self.actions)
        r = any(fnmatch.fnmatch(resource, pat) for pat in self.resources)
        return a and r


class Policy:
    """A policy document evaluated with AWS rules: explicit Deny wins; else Allow if any statement allows; else
    implicit deny. `evaluate` returns 'Deny' (explicit), 'Allow', or 'Implicit'."""

    def __init__(self, statements):
        self.statements = list(statements)

    def evaluate(self, action: str, resource: str) -> str:
        applicable = [s for s in self.statements if s.matches(action, resource)]
        if any(s.effect == "Deny" for s in applicable):
            return "Deny"
        if any(s.effect == "Allow" for s in applicable):
            return "Allow"
        return "Implicit"


class ComposedAuthority:
    """ReDevOps composes an organizational guardrail (SCP), the role's IAM policy, and the Mission grant — with
    AWS deny-wins semantics. An action is admitted only when: no explicit Deny anywhere, IAM allows it, and the
    Mission grants it. A permissive IAM Allow never overrides an explicit SCP Deny."""

    def __init__(self, scp: Policy, iam: Policy, mission_grants):
        self.scp = scp
        self.iam = iam
        self.mission = frozenset(mission_grants)

    def decision(self, action: str, resource: str) -> tuple[bool, str]:
        if self.scp.evaluate(action, resource) == "Deny":
            return False, "SCP guardrail explicitly denies this action (deny-wins)"
        if self.iam.evaluate(action, resource) == "Deny":
            return False, "IAM policy explicitly denies this action (deny-wins)"
        if self.iam.evaluate(action, resource) != "Allow":
            return False, "no IAM Allow for this action/resource (implicit deny)"
        if action not in self.mission:
            return False, "the Mission does not grant this action (default-deny)"
        return True, "admitted"


class StrandsAgentCapability:
    """WorkflowAdapter over a real Strands Agents Agent (OpenAI-compatible model)."""

    framework = "strands-agents"
    framework_version = STRANDS_VERSION

    def __init__(self, *, name: str = "strands_agent", system_prompt: str = "Answer concisely.",
                 model: str = "gpt-4o-mini"):
        self._name = name
        self._system = system_prompt
        self.model_id = model
        self._agent = None
        self._last_span: NodeSpan | None = None

    def _build(self):
        from strands import Agent
        from strands.models.openai import OpenAIModel
        m = OpenAIModel(client_args={"api_key": os.environ["OPENAI_API_KEY"],
                                     "base_url": os.environ.get("OPENAI_BASE_URL")}, model_id=self.model_id)
        self._agent = Agent(model=m, system_prompt=self._system, callback_handler=None)
        return self._agent

    def run(self, prompt: str) -> str:
        agent = self._agent or self._build()
        return str(agent(prompt)).strip()

    def invoke(self, capability: str, inputs: dict) -> dict:
        answer = self.run(inputs.get("prompt") or inputs.get("question") or "")
        self._last_span = self.telemetry()
        return {"answer": answer, "framework": self.framework, "model": self.model_id}

    def node_identity(self, capability: str, inputs: dict) -> str:
        h = hashlib.sha256((capability + str(sorted(inputs.items()))).encode()).hexdigest()[:10]
        return f"strands:{self._name}:{h}"

    def state_projection(self) -> dict:
        return {"framework": self.framework, "agent": self._name, "model": self.model_id}

    def telemetry(self) -> NodeSpan:
        return NodeSpan(node_id=f"strands:{self._name}", kind="agent", name=f"strands:{self._name}",
                        attrs={"framework": self.framework},
                        children=[NodeSpan(node_id="model", kind="model", name=self.model_id)])
