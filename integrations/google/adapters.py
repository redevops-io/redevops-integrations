"""Google ADK / A2A adapters for ReDevOps (frameworks plan Section 7 / hyperscaler plan).

Wraps a REAL Google Agent Development Kit (`google-adk`) agent — an `Agent` driven by `LiteLlm` against any
OpenAI-compatible endpoint — as a ReDevOps capability, and models an A2A delegation mesh. Google ADK/A2A keep
owning agent construction, the agent loop, and agent-to-agent delegation; ReDevOps adds — the A2A-specific
boundary — an authority envelope that narrows across EVERY hop of a multi-hop mesh (transitive attenuation) and a
provenance requirement so a coordinator only admits findings attributable to authorized agents.

Runs entirely locally (ADK + LiteLlm point at a local endpoint) — no Google Cloud credentials. Coexists in the
shared benchmark venv (installing google-adk left pydantic/numpy/torch untouched).
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))

from adapter import NodeSpan  # noqa: E402

try:
    import google.adk as _adk
    ADK_VERSION = getattr(_adk, "__version__", "2.8.0")
except Exception:
    ADK_VERSION = "2.8.0"

os.environ.setdefault("OPENAI_MODEL_NAME", "gpt-4o-mini")


class AuthorityChain:
    """Authority across a multi-hop A2A delegation mesh. A hop's effective grants are the INTERSECTION of every
    declared grant from the root down to that hop — so authority is monotone non-increasing across the whole
    chain, not just the first handoff. A 3rd-hop agent can never exceed what an intermediate hop was granted."""

    def __init__(self, hops):
        self.hops = list(hops)                       # [(agent, {grants}), ...] root first

    def effective(self) -> frozenset:
        eff = None
        for _agent, grants in self.hops:
            g = frozenset(grants)
            eff = g if eff is None else (eff & g)
        return eff if eff is not None else frozenset()

    def extend(self, agent, grants) -> "AuthorityChain":
        return AuthorityChain(self.hops + [(agent, grants)])

    def admits(self, permission: str) -> bool:
        return permission in self.effective()

    def path(self) -> str:
        return " → ".join(a for a, _ in self.hops)


class Provenance:
    """The ordered chain of agents that produced/relayed a finding. A finding is attributable only when every
    agent in its chain is a known, authorized mesh agent — a forged or broken chain is not admissible."""

    def __init__(self, chain):
        self.chain = list(chain)

    def attributable(self, authorized_agents) -> bool:
        return bool(self.chain) and all(a in authorized_agents for a in self.chain)

    def digest(self) -> str:
        return "sha256:" + hashlib.sha256("→".join(self.chain).encode()).hexdigest()[:12]


class AdkAgentCapability:
    """WorkflowAdapter over a real google-adk Agent (LiteLlm). Usable as a Mission capability handler."""

    framework = "google-adk"
    framework_version = ADK_VERSION

    def __init__(self, *, name: str = "adk_agent", instruction: str = "Answer concisely.",
                 model: str = "gpt-4o-mini"):
        self._name = name
        self._instruction = instruction
        self.model_id = model
        self._agent = None
        self._last_span: NodeSpan | None = None

    def _build(self):
        from google.adk.agents import Agent
        from google.adk.models.lite_llm import LiteLlm
        self._agent = Agent(name=self._name, model=LiteLlm(model=f"openai/{self.model_id}"),
                            instruction=self._instruction)
        return self._agent

    async def _arun(self, prompt: str) -> str:
        from google.adk.runners import InMemoryRunner
        from google.genai import types
        agent = self._agent or self._build()
        runner = InMemoryRunner(agent=agent, app_name="mesh")
        await runner.session_service.create_session(app_name="mesh", user_id="u", session_id="s")
        msg = types.Content(role="user", parts=[types.Part(text=prompt)])
        out = ""
        async for ev in runner.run_async(user_id="u", session_id="s", new_message=msg):
            if ev.is_final_response() and ev.content:
                out = "".join(p.text or "" for p in ev.content.parts)
        return out.strip()

    def run(self, prompt: str) -> str:
        return asyncio.run(self._arun(prompt))

    def invoke(self, capability: str, inputs: dict) -> dict:
        answer = self.run(inputs.get("prompt") or inputs.get("question") or "")
        self._last_span = self.telemetry()
        return {"answer": answer, "framework": self.framework, "model": self.model_id}

    def node_identity(self, capability: str, inputs: dict) -> str:
        h = hashlib.sha256((capability + str(sorted(inputs.items()))).encode()).hexdigest()[:10]
        return f"google-adk:{self._name}:{h}"

    def state_projection(self) -> dict:
        return {"framework": self.framework, "agent": self._name, "model": self.model_id}

    def telemetry(self) -> NodeSpan:
        return NodeSpan(node_id=f"google-adk:{self._name}", kind="agent", name=f"adk:{self._name}",
                        attrs={"framework": self.framework},
                        children=[NodeSpan(node_id="model", kind="model", name=self.model_id)])
