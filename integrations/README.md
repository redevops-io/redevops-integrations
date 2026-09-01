# ReDevOps × Agent Frameworks / Hyperscalers — reference implementations

End-to-end integrations proving ReDevOps adds measurable production-runtime value **around** an existing agent
framework or managed cloud stack, without replacing it. Implements the two blueprints
(`REDEVOPS_OPEN_AGENT_FRAMEWORKS_END_TO_END_PLAN_v2.md`, `REDEVOPS_HYPERSCALER_END_TO_END_REFERENCE_IMPLEMENTATIONS_v2.md`).

- **`common/`** — provider-neutral harness reused by every integration: the result-classification enum
  (`PARITY / FRAMEWORK_NATIVE_ADVANTAGE / REDEVOPS_DELTA / EXPECTED_IMPLEMENTATION_DIFFERENCE / BUG`), the
  **Workflow Adapter Contract** (`adapter.py`), and the reproducibility **AcceptanceBundle** (`bundle.py`).
- **`nvidia/`** — **DONE.** NeMo Agent Toolkit × ReDevOps on SciFact + GDPR Closure Resolution + governed tool.
  Real `nvidia-nat` agent; 10 REDEVOPS_DELTA / 1 EXPECTED_IMPLEMENTATION_DIFFERENCE / 0 BUG. See `nvidia/README.md`.

Planned next (same `common/` contract, so each is a conformance exercise): PydanticAI (RAAAL verified-intent),
LlamaIndex Workflows (long-context + code closure), CrewAI (multi-agent business workflow); then the hyperscaler
managed stacks (AWS Wealth-Manager shadow, Google ADK/A2A research mesh, Azure governed deployment, DigitalOcean
shared-Runtime multi-app). A small provider-neutral Mission is reserved for **semantic conformance only**, never
framework ranking.

Runs in `redevops-integrations/.venv` (which now also has `nvidia-nat`).
