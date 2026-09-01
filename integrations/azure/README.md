# Azure (Microsoft Semantic Kernel) × ReDevOps — reference implementation

> **Semantic Kernel owns the agent loop and function-calling. ReDevOps turns a deployment into a governed Mission —
> a compliance-control gate, Entra-composing authority (deny-wins), approval and replay.**
>
> A real `semantic-kernel` **1.36.0** agent runs **locally, in-process** with the Mission SDK — auto function-calling
> against any OpenAI-compatible endpoint. **No Azure cloud credentials required.** No mock.

This follows the reference template: **reference implementation → benchmark evidence → architecture → reproducible
example.** It is framed as *conformance + incremental runtime value*, never a "Semantic Kernel vs ReDevOps" scorecard.

**Runs entirely on your own hardware.** Microsoft's agent framework is an open-source SDK; the agent loop and
function-calling run in-process, and the only outbound calls are model calls, which point at any OpenAI-compatible
endpoint (a local proxy here). Azure Entra identity is represented as a policy object, not a live cloud call — so
the governance semantics are exercised for real without an Azure subscription.

**The differentiation for Azure is precise.** Semantic Kernel's strongest abstraction is function-calling: the
agent plans and invokes plugin functions. So we test the boundary *immediately outside* it. The agent will invoke
`request_deploy(service, environment)` whenever it decides to — it has no notion of whether the target is
*compliant* or the identity is *authorized*. ReDevOps admits the deployment only when every required compliance
control for the environment is evidenced (deny-by-default) **and** the identity's Entra role permits it (authority
composes as **deny-wins**), then runs it as a Mission with approval, exactly-once execution and replay.

## Architecture

```
Semantic Kernel                     what Semantic Kernel owns
  Kernel · plugins / functions        the agent loop + function-calling
  planners                            plan construction
        │
        ▼
ReDevOps Runtime                   what ReDevOps adds (around it)
  Compliance-control gate            deny-by-default until every required control is evidenced
  Authority (Entra deny-wins)        effective authority = framework grant ∩ Entra role
  Approval · Replay                  exactly-once deploy, event-sourced replay
```

## Install (dedicated venv)

Semantic Kernel pins `pydantic`/`openai` **below** the shared benchmark environment (which is held at the CrewAI
floor of pydantic 2.12.5). Forcing it into the shared venv would ratchet the four framework slices down again, so
this integration uses its **own venv** — the same provider-neutral `common/` contract, isolated dependencies.

```bash
uv venv integrations/azure/.venv --python 3.12
VIRTUAL_ENV=integrations/azure/.venv uv pip install semantic-kernel -e /mnt/backup/projects/mission-sdk
```

## Run

```bash
cd integrations/azure
.venv/bin/python conformance.py                          # AZ-A..E + classification tally
.venv/bin/python examples/compliance_gated_deploy.py     # AZ-A: compliant+authorized deploys only (real SK, cached)
.venv/bin/python examples/governed_deploy.py             # AZ-B: the deploy executes as a governed Mission
```

## Expected output (the measured proof)

```
AZ-A compliance gate    native runs 3/3 non-compliant/unauthorized deploys · ReDevOps denies all 3 (0 wrong)
                        compliant + authorized deployments admitted 2/2
AZ-B governed deploy    0 unauthorized · 0 before approval · exactly 1 after · 0 duplicates on replay
AZ-C deny-wins          framework grant {staging,production} ∩ Entra {staging} = staging; prod denied though compliant
AZ-E telemetry          mission → agent → function → model (Mission is the causal root)
conformance             7 REDEVOPS_DELTA · 1 PARITY · 0 EXPECTED_IMPLEMENTATION_DIFFERENCE · 0 BUG
```

**Honesty line:** AZ-A's compliance/authority gate is deterministic — the value is the enforced mechanism, not a
model score. The real Semantic Kernel agent genuinely plans and invokes the deployment function in every case; what
ReDevOps changes is whether that request is *allowed to become a deployment*.

## Acceptance bundle (reproducibility)

```
results/az_a_compliance_gated_deploy.json   results/az_b_governed_deploy.json
```

## Layout

```
integrations/
  common/                 # provider-neutral harness (shared with nvidia/, pydanticai/, llamaindex/, crewai/)
  azure/
    .venv/                   # dedicated env (Semantic Kernel's dependency floor; gitignored)
    adapters.py              # KernelAgentCapability (real SK agent) · DeploymentAdmission (compliance + Entra deny-wins)
    examples/                # compliance_gated_deploy (AZ-A) · governed_deploy (AZ-B)
    conformance.py           # runs AZ-A..E, prints the classification tally
    results/                 # emitted acceptance bundles
```

Same `common/` adapter contract as the four framework slices — new provider support is a conformance exercise, not
a new architecture. This is the first slice that runs in a dedicated venv, purely to isolate Semantic Kernel's
dependency floor.
