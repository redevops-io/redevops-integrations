"""Azure (Microsoft Semantic Kernel) adapters for ReDevOps (frameworks plan Section 7 / hyperscaler plan).

Wraps a REAL Semantic Kernel agent — a `Kernel` with plugin functions, driven by auto function-calling — as a
ReDevOps capability. Semantic Kernel keeps owning the agent loop, plugins/functions, and planning; ReDevOps adds a
governed deployment Mission around it: a compliance-control gate (deny-by-default on any unsatisfied control),
environment authority that composes with an Entra role as DENY-WINS, plus approval, exactly-once and replay.

Runs entirely locally (Semantic Kernel points at any OpenAI-compatible endpoint) — no Azure cloud credentials.
Because Semantic Kernel pins pydantic/openai below the shared benchmark env, this integration uses a dedicated
venv: `integrations/azure/.venv`.
"""
from __future__ import annotations

import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))

from adapter import NodeSpan  # noqa: E402

try:
    import semantic_kernel as _sk
    SK_VERSION = getattr(_sk, "__version__", "1.36.0")
except Exception:
    SK_VERSION = "1.36.0"


# ---- ReDevOps deployment admission (compliance controls + environment authority) ----
class Control:
    """A single compliance control the deployment must satisfy, with evidence (Control / ControlAssertion)."""

    def __init__(self, control_id: str, description: str):
        self.control_id = control_id
        self.description = description


# the controls a governed deployment must evidence, per target environment
REQUIRED_CONTROLS = {
    "staging": ["CTL-ENCRYPTION", "CTL-IMAGE"],
    "production": ["CTL-ENCRYPTION", "CTL-IMAGE", "CTL-NETWORK", "CTL-CHANGE-APPROVED"],
}


class EntraRole:
    """A simulated Entra ID role assignment — the environments an identity is permitted to deploy to."""

    def __init__(self, principal: str, allowed_envs):
        self.principal = principal
        self.allowed = frozenset(allowed_envs)


class DeploymentAdmission:
    """Deny-by-default admission for a deployment: every REQUIRED control for the target env must be evidenced, AND
    the identity's Entra role must permit the env. Authority composes as DENY-WINS — a permissive framework grant
    never overrides a restrictive Entra role."""

    def __init__(self, entra: EntraRole, framework_grant_envs=("staging", "production")):
        self.entra = entra
        self.framework = frozenset(framework_grant_envs)

    def authorized_envs(self) -> frozenset:
        return self.entra.allowed & self.framework          # deny-wins: intersection, not union

    def admit(self, env: str, evidenced_controls) -> tuple[bool, str]:
        if env not in self.authorized_envs():
            return False, f"authority: Entra role for {self.entra.principal} does not permit {env} (deny-wins)"
        missing = [c for c in REQUIRED_CONTROLS.get(env, []) if c not in set(evidenced_controls)]
        if missing:
            return False, f"compliance: unsatisfied controls for {env}: {', '.join(missing)}"
        return True, "admitted"


class KernelAgentCapability:
    """WorkflowAdapter over a real Semantic Kernel agent (Kernel + plugin + auto function-calling)."""

    framework = "semantic-kernel"
    framework_version = SK_VERSION

    def __init__(self, *, name: str = "sk_agent", model: str = "gpt-4o-mini"):
        self._name = name
        self.model_id = model
        self._kernel = None
        self._calls: list = []
        self._last_span: NodeSpan | None = None

    def _build(self):
        import semantic_kernel as sk
        from openai import AsyncOpenAI
        from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
        from semantic_kernel.functions import kernel_function

        client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ.get("OPENAI_BASE_URL"))
        k = sk.Kernel()
        k.add_service(OpenAIChatCompletion(ai_model_id=self.model_id, async_client=client, service_id="oai"))
        calls = self._calls

        class DeploymentPlugin:
            @kernel_function(name="request_deploy", description="Request a deployment of a service to an environment")
            def request_deploy(self, service_name: str, environment: str) -> str:
                calls.append({"service": service_name, "environment": environment})
                return f"deployment requested: {service_name} -> {environment}"

        k.add_plugin(DeploymentPlugin(), "ops")
        self._kernel = k
        return k

    async def _aplan(self, prompt: str) -> dict:
        import semantic_kernel as sk  # noqa: F401
        from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
        from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings
        from semantic_kernel.contents import ChatHistory

        k = self._kernel or self._build()
        self._calls.clear()
        settings = OpenAIChatPromptExecutionSettings(service_id="oai")
        settings.function_choice_behavior = FunctionChoiceBehavior.Auto()
        svc = k.get_service("oai")
        hist = ChatHistory()
        hist.add_user_message(prompt)
        await svc.get_chat_message_content(chat_history=hist, settings=settings, kernel=k)
        return self._calls[0] if self._calls else {}

    def plan(self, prompt: str) -> dict:
        import asyncio
        return asyncio.run(self._aplan(prompt))

    def invoke(self, capability: str, inputs: dict) -> dict:
        req = self.plan(inputs.get("prompt") or "")
        self._last_span = self.telemetry()
        return {"request": req, "framework": self.framework, "model": self.model_id}

    def node_identity(self, capability: str, inputs: dict) -> str:
        h = hashlib.sha256((capability + str(sorted(inputs.items()))).encode()).hexdigest()[:10]
        return f"semantic-kernel:{self._name}:{h}"

    def state_projection(self) -> dict:
        return {"framework": self.framework, "agent": self._name, "model": self.model_id}

    def telemetry(self) -> NodeSpan:
        return NodeSpan(node_id=f"semantic-kernel:{self._name}", kind="agent", name=f"sk:{self._name}",
                        attrs={"framework": self.framework},
                        children=[NodeSpan(node_id="function", kind="tool", name="ops.request_deploy",
                                           children=[NodeSpan(node_id="model", kind="model", name=self.model_id)])])
