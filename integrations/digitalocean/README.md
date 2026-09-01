# DigitalOcean (GenAI Platform) × ReDevOps — reference implementation

> **DigitalOcean gives several lean apps a managed agent stack. ReDevOps gives all of them ONE runtime —
> tenant-isolated authority, context, cache and replay — so no app rebuilds production machinery.**
>
> Three real DigitalOcean GenAI app-agents (OpenAI-compatible) run **locally, in-process** on one Mission SDK
> runtime. **No DigitalOcean credentials.** No mock.

This follows the reference template: **reference implementation → benchmark evidence → architecture → reproducible
example.** It is framed as *conformance + incremental runtime value*, never a "DigitalOcean vs ReDevOps" scorecard.

**The differentiation for DigitalOcean is the shared-Runtime thesis, not a new framework.** DigitalOcean's GenAI
Platform is an OpenAI-compatible managed agent + knowledge-base stack, ideal for lean apps. The gap it leaves is
the one this whole series is about: every app still rebuilds context, retry, replay, governance and telemetry. So
the DO slice runs three real apps — **telegrambot.ai · nutrients.tech · vibexgen.io** — on ONE ReDevOps Runtime,
which provides that machinery once. The boundary immediately outside a naive shared runtime is **tenant
isolation**: one app must not read another's context or exercise another's authority. ReDevOps scopes every
access to the acting tenant, deny-by-default across tenants, with tenant-namespaced cache keys.

## Architecture

```
DigitalOcean GenAI Platform         what DigitalOcean owns
  managed agent + knowledge base      the OpenAI-compatible agent stack, per app
        │
        ▼
ReDevOps Runtime (ONE, shared)     what ReDevOps adds (around all the apps)
  Tenant isolation                   own-namespace reads only; deny cross-tenant; tenant-scoped cache keys
  Context · retry · replay           the production machinery, provided once instead of per app
  Authority · Approval · Governance  each tenant's consequential action governed independently
```

## Install

Runs locally, no DigitalOcean cloud. Nothing to install beyond the shared benchmark venv — DO GenAI is
OpenAI-compatible, so a DO agent is an OpenAI client (pointed at a local endpoint here; swap `base_url` for a real
DO inference endpoint).

## Run

```bash
cd integrations/digitalocean
../../.venv/bin/python conformance.py                    # DO-A..E + classification tally
../../.venv/bin/python examples/shared_runtime.py        # DO-A: three apps, tenant-isolated (no LLM)
../../.venv/bin/python examples/governed_tenant.py       # DO-B: a tenant's publish, governed as a Mission
../../.venv/bin/python examples/three_apps.py            # DO-C: three real DO agents on one Runtime (cached)
```

## Expected output (the measured proof)

```
DO-A shared runtime     cross-tenant: native runs 3/3 · ReDevOps denies all · in-tenant 3/3 admitted
                        shared machinery: 5 provided once vs 15 rebuilt per app (3 apps × 5 capabilities)
DO-B governed action    0 unauthorized · 0 before approval · exactly 1 after · 0 duplicates on replay
DO-C three real apps    telegrambot + nutrients + vibexgen ran on one runtime · 3/3 tenant-isolated
DO-D cache isolation    the same question from two apps yields distinct cache keys (no cross-tenant bleed)
DO-E telemetry          mission → {app} → model (the one shared Mission is the causal root)
conformance             9 REDEVOPS_DELTA · 2 PARITY · 0 EXPECTED_IMPLEMENTATION_DIFFERENCE · 0 BUG
```

**Honesty line:** the isolation and cache-key mechanics are deterministic — the value is the enforced boundary.
The three app-agents in DO-C are real OpenAI-compatible calls; DigitalOcean GenAI identity/knowledge-base wiring is
represented by tenant scopes rather than a live DO account. The shared-machinery count is a capability tally, not a
performance number.

## Acceptance bundle (reproducibility)

```
results/do_a_shared_runtime.json   results/do_b_governed_tenant.json   results/do_c_three_apps.json
```

## Layout

```
integrations/
  common/                 # provider-neutral harness (shared with every framework + hyperscaler slice)
  digitalocean/
    adapters.py              # DoAgentCapability (real OpenAI-compatible agent) · TenantScope · SharedRuntime
    examples/                # shared_runtime (DO-A) · governed_tenant (DO-B) · three_apps (DO-C)
    conformance.py           # runs DO-A..E, prints the classification tally
    results/                 # emitted acceptance bundles
```

Same `common/` adapter contract as every other slice. This is the shared-Runtime slice: the differentiator is one
runtime under several apps, not a new agent framework.
