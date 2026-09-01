"""DigitalOcean (GenAI Platform) adapters for ReDevOps (frameworks plan Section 7 / hyperscaler plan).

DigitalOcean's GenAI Platform exposes an OpenAI-compatible agent API, so a real DO agent is an OpenAI client
pointed at a DO inference endpoint (a local endpoint stands in here — no DO credentials). Unlike the other slices,
the DO differentiator is not a new agent framework: it is the SHARED RUNTIME thesis. DigitalOcean gives several
lean apps a managed agent + knowledge-base stack; ReDevOps gives all of those apps ONE runtime — tenant-isolated
authority, context and replay — so no app rebuilds production machinery, and no app can reach into another's state.

Represents the real DO workload: telegrambot.ai · nutrients.tech · vibexgen.io sharing one ReDevOps Runtime.
"""
from __future__ import annotations

import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))

from adapter import NodeSpan  # noqa: E402

DO_API = "openai-compatible/v1"          # DO GenAI Platform is OpenAI-compatible


class TenantScope:
    """A tenant app on the shared Runtime — its id, the grants it may exercise, and its private context namespace.
    Isolation is deny-by-default across tenants: an app may only act within its own grants and read its own
    namespace. A shared runtime with no tenant scope lets any app reach any state."""

    def __init__(self, tenant_id: str, grants, namespace: str | None = None):
        self.tenant_id = tenant_id
        self.grants = frozenset(grants)
        self.namespace = namespace or f"tenant/{tenant_id}"

    def may_act(self, action: str) -> bool:
        return action in self.grants

    def may_read(self, resource_namespace: str) -> bool:
        return resource_namespace == self.namespace

    def cache_key(self, key: str) -> str:
        # tenant-scoped cache key — the fix for cross-tenant cache bleed
        return f"{self.tenant_id}::{key}"


class SharedRuntime:
    """One ReDevOps Runtime shared by several DO apps. It provides the production capabilities each app would
    otherwise rebuild — context, retry, replay, governance, telemetry — once, and mediates every tenant access."""

    CAPABILITIES = ("context", "retry", "replay", "governance", "telemetry")

    def __init__(self, tenants):
        self.tenants = {t.tenant_id: t for t in tenants}

    def access(self, acting_tenant: str, action: str, resource_namespace: str) -> tuple[bool, str]:
        t = self.tenants.get(acting_tenant)
        if t is None:
            return False, f"unknown tenant {acting_tenant}"
        if not t.may_read(resource_namespace):
            return False, f"cross-tenant read denied: {acting_tenant} → {resource_namespace}"
        if not t.may_act(action):
            return False, f"{acting_tenant} is not granted {action}"
        return True, "admitted"

    def machinery_rebuilt_per_app(self) -> int:
        return len(self.tenants) * len(self.CAPABILITIES)     # what N apps would each rebuild

    def machinery_provided_once(self) -> int:
        return len(self.CAPABILITIES)                          # what the one runtime provides


class DoAgentCapability:
    """WorkflowAdapter over a real OpenAI-compatible DO GenAI agent, scoped to one tenant app."""

    framework = "digitalocean-genai"
    framework_version = DO_API

    def __init__(self, *, tenant: str, name: str = "do_agent", system_prompt: str = "Answer concisely.",
                 model: str = "gpt-4o-mini"):
        self.tenant = tenant
        self._name = name
        self._system = system_prompt
        self.model_id = model
        self._last_span: NodeSpan | None = None

    def run(self, prompt: str) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ.get("OPENAI_BASE_URL"))
        r = client.chat.completions.create(
            model=self.model_id, max_tokens=60,
            messages=[{"role": "system", "content": self._system}, {"role": "user", "content": prompt}])
        return (r.choices[0].message.content or "").strip()

    def invoke(self, capability: str, inputs: dict) -> dict:
        answer = self.run(inputs.get("prompt") or inputs.get("question") or "")
        self._last_span = self.telemetry()
        return {"answer": answer, "tenant": self.tenant, "framework": self.framework, "model": self.model_id}

    def node_identity(self, capability: str, inputs: dict) -> str:
        h = hashlib.sha256((self.tenant + capability + str(sorted(inputs.items()))).encode()).hexdigest()[:10]
        return f"do:{self.tenant}:{self._name}:{h}"

    def state_projection(self) -> dict:
        return {"framework": self.framework, "tenant": self.tenant, "agent": self._name, "model": self.model_id}

    def telemetry(self) -> NodeSpan:
        return NodeSpan(node_id=f"do:{self.tenant}:{self._name}", kind="agent", name=f"do:{self._name}",
                        attrs={"framework": self.framework, "tenant": self.tenant},
                        children=[NodeSpan(node_id="model", kind="model", name=self.model_id)])
