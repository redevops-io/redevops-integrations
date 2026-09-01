# NVIDIA NeMo Agent Toolkit × ReDevOps — reference implementation

> **NeMo owns the agent loop. ReDevOps adds closure-aware context and governed Mission semantics.**
>
> A real NVIDIA NeMo Agent Toolkit **v1.8.0** agent runs **in-process** with the Mission SDK and Closure
> Resolution. No mock, no sidecar approximation.

This is the reference every later framework/provider integration follows: **reference implementation → benchmark
evidence → architecture → reproducible example.** It is framed as *conformance + incremental runtime value*, never
a "NeMo vs ReDevOps" scorecard.

- **View the reference implementation** — this directory.
- **Run the benchmark** — `../../.venv/bin/python conformance.py` (below).
- **Read the architecture article** — redevops.io/blog/nemo-agent-toolkit-meets-redevops-runtime.

## Architecture

```
NVIDIA NeMo Agent Toolkit          what NeMo owns
  agent loop · tools · models        the agent construction + execution
  evaluator · telemetry              native optimization + evaluation
        │
        ▼
ReDevOps Runtime                   what ReDevOps adds (around it)
  Mission · Context / Closure        durable objective + closure-aware context
  Authority · Replay                 approval, exactly-once, event-sourced replay
  Verification · Governance          resolve-or-abstain, deny-by-default
```

ReDevOps does **not** re-implement NeMo's harness, model/tool access, native optimization, or evaluation. The
NeMo agent's own retrieval can even be registered as a Context-Runtime representation the optimizer may *choose*.

## Install

Everything is one Python env (`redevops-integrations/.venv`): the Mission SDK, the Closure Resolution stack, and
`nvidia-nat` coexist — installing nat only downgraded `pandas` (torch/numpy/pydantic untouched).

```bash
VIRTUAL_ENV=.venv uv pip install nvidia-nat      # already present in redevops-integrations/.venv
```

## Run

```bash
cd integrations/nvidia
../../.venv/bin/python conformance.py               # NV-A..E + classification tally
../../.venv/bin/python examples/scifact_closure.py  # NV-A: NeMo verification, naive vs closure (gpt-4o-mini, cached)
../../.venv/bin/python examples/gdpr_structural.py  # NV-B: GDPR structural closure (no LLM)
../../.venv/bin/python examples/governed_tool.py    # NV-D: governed action, approval + replay (real NeMo proposer)
```

## Expected output (the measured proof)

```
NV-A SciFact   verification accuracy naive 0.800 → closure 0.867 · evidence recall 0.841 → 0.877
               selective autonomy 0.857 @ 93% coverage
NV-B GDPR      content-only 0.412 → composed (content+graph) 0.913 · seed-only 0.000 → 0.913
NV-D governed  0 unauthorized · 0 side-effects before approval · exactly 1 after · 0 duplicates on replay
NV-E telemetry mission → agent → model (Mission is the causal root)
conformance    10 REDEVOPS_DELTA · 1 EXPECTED_IMPLEMENTATION_DIFFERENCE · 0 BUG
```

**Honesty line:** SciFact is already content-saturated, so the closure gain is modest (+0.067); GDPR is
structural, so closure adds much more (+0.501). This is *incremental runtime value*, not a framework comparison.

## Acceptance bundle (reproducibility)

Each run writes a reproducibility record to `results/` — framework version, model ids, dataset hashes, per-finding
classification, and a `result_digest`:

```
results/nv_a_scifact_closure.json     results/nv_b_gdpr_structural.json     results/nv_d_governed_tool.json
```

## Layout

```
integrations/
  common/                 # provider-neutral harness (reused by every framework/cloud)
    classification.py        # PARITY / FRAMEWORK_NATIVE_ADVANTAGE / REDEVOPS_DELTA / EXPECTED_… / BUG
    adapter.py               # ReDevOps Workflow Adapter Contract (invoke · identity · state · telemetry)
    bundle.py                # AcceptanceBundle reproducibility record
  nvidia/
    adapters.py              # NemoAgentCapability (wraps a real nat Function) + Evaluator/Telemetry adapters
    examples/                # scifact_closure (NV-A) · gdpr_structural (NV-B) · governed_tool (NV-D)
    conformance.py           # runs NV-A..E, prints the classification tally
    results/                 # emitted acceptance bundles
```

Future frameworks (PydanticAI · LlamaIndex · CrewAI) and hyperscalers (AWS · Google · Azure · DigitalOcean)
implement the **same** `common/` adapter contract — new support is a conformance exercise, not a new architecture.
