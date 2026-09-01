"""NVIDIA NeMo Agent Toolkit adapters for ReDevOps (both plans, Section 5/6).

Wraps a REAL `nvidia-nat` (NeMo Agent Toolkit) Function/Workflow as a ReDevOps Capability so a Mission node can
invoke it, and nests NeMo's native telemetry UNDER the Mission causal trace. The NeMo agent keeps owning the
agent loop, model access, and evaluation; ReDevOps adds Closure Resolution, authority, replay, verification and
governance around it — without duplicating NeMo-native capabilities.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "workloads", "legal_change_closure"))

from adapter import NodeSpan  # noqa: E402

# real NeMo Agent Toolkit
from nat.builder.function_info import FunctionInfo  # noqa: E402
from nat.builder.workflow_builder import WorkflowBuilder  # noqa: E402
from nat.cli.register_workflow import register_function  # noqa: E402
from nat.data_models.function import FunctionBaseConfig  # noqa: E402

import legal_llm as _llm  # noqa: E402  (OpenAI-compatible chat helper: openai / local Qwen)

try:
    import nat as _nat
    NAT_VERSION = getattr(_nat, "__version__", "1.8.0")
except Exception:
    NAT_VERSION = "1.8.0"


class RedevopsAgentConfig(FunctionBaseConfig, name="redevops_nemo_agent"):
    """A NeMo Agent Toolkit function config — the agent NeMo owns."""
    system: str = "You are a careful analyst. Answer concisely."
    provider: str = "openai"          # openai (gpt-4o-mini) | local (Qwen)
    max_tokens: int = 400


@register_function(config_type=RedevopsAgentConfig)
async def _build_redevops_agent(config: RedevopsAgentConfig, builder):
    if config.provider == "openai":
        os.environ.setdefault("CR_OPENAI_MODEL", "gpt-4o-mini")

    async def _fn(prompt: str) -> str:
        return _llm.chat(config.system, prompt, max_tokens=config.max_tokens, provider=config.provider)

    yield FunctionInfo.from_fn(_fn, description="ReDevOps-wrapped NeMo research/analysis agent")


class NemoAgentCapability:
    """WorkflowAdapter over a real nat Function. Usable directly as a Mission capability handler (dict->dict)."""

    framework = "nvidia-nemo-agent-toolkit"
    framework_version = NAT_VERSION

    def __init__(self, system: str, *, provider: str = "openai", name: str = "nemo_agent", max_tokens: int = 400):
        self._cfg = RedevopsAgentConfig(system=system, provider=provider, max_tokens=max_tokens)
        self._name = name
        self.model_id = "gpt-4o-mini" if provider == "openai" else "qwen-local"
        self._last_span: NodeSpan | None = None

    async def _abuild_invoke(self, prompt: str) -> str:
        async with WorkflowBuilder() as b:
            fn = await b.add_function(self._name, self._cfg)
            return await fn.ainvoke(prompt, to_type=str)

    def invoke(self, capability: str, inputs: dict) -> dict:
        prompt = inputs.get("prompt") or inputs.get("question") or ""
        answer = asyncio.run(self._abuild_invoke(prompt))
        self._last_span = NodeSpan(
            node_id=self.node_identity(capability, inputs), kind="agent",
            name=f"nat:{self._name}", attrs={"framework": self.framework, "version": self.framework_version},
            children=[NodeSpan(node_id="model", kind="model", name=self.model_id, attrs={"provider": self._cfg.provider})])
        return {"answer": answer, "framework": self.framework, "model": self.model_id}

    def node_identity(self, capability: str, inputs: dict) -> str:
        h = hashlib.sha256((capability + str(sorted(inputs.items()))).encode()).hexdigest()[:10]
        return f"nat:{self._name}:{h}"

    def state_projection(self) -> dict:
        return {"framework": self.framework, "agent": self._name, "model": self.model_id}

    def telemetry(self) -> NodeSpan:
        return self._last_span or NodeSpan(node_id=self._name, kind="agent", name=f"nat:{self._name}")


class NemoEvaluatorAdapter:
    """Maps a scoring function onto a NeMo-evaluator-style result (accuracy over N items)."""

    def __init__(self, name: str = "closure_accuracy"):
        self.name = name

    def evaluate(self, predictions, golds) -> dict:
        n = len(golds)
        correct = sum(1 for p, g in zip(predictions, golds) if p == g)
        return {"evaluator": self.name, "n": n, "accuracy": (correct / n) if n else 0.0, "correct": correct}


class NemoTelemetryAdapter:
    """Nests native NeMo spans under a Mission node (never promotes a substrate span to Mission root)."""

    @staticmethod
    def nest(mission_node_id: str, agent_span: NodeSpan) -> NodeSpan:
        return NodeSpan(node_id=mission_node_id, kind="workflow_step", name="mission.node",
                        children=[agent_span])
