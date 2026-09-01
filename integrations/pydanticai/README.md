# PydanticAI × ReDevOps — reference implementation

> **PydanticAI owns the typed agent loop. ReDevOps supplies the runtime properties that structure alone doesn't establish.**
>
> A real `pydantic-ai` **v2.36.0** typed `Agent` runs **in-process** with the Mission SDK, Closure Resolution,
> and the governance/replay layer. No mock, no sidecar approximation.

This follows the reference template: **reference implementation → benchmark evidence → architecture → reproducible
example.** It is framed as *conformance + incremental runtime value*, never a "PydanticAI vs ReDevOps" scorecard.

**The differentiation for PydanticAI is precise.** PydanticAI's typed boundary guarantees an output's *structure* —
a well-typed, schema-valid Pydantic model. Structure alone does not establish that the output is *supported by
evidence*, *fresh*, *authorized*, or *policy-admissible*, nor that a restart won't act on it twice. An application
can of course build some of those checks itself; the point is that they are not properties of the type, and ReDevOps
supplies them once as shared execution infrastructure — verification, governance and replay, orthogonal to schema
validity. This integration makes that orthogonality measurable.

## Architecture

```
PydanticAI                          what PydanticAI owns
  typed Agent · output_type          the agent loop + validated typed outputs
  deps · pydantic-graph              typed dependencies + local control flow
        │
        ▼
ReDevOps Runtime                   what ReDevOps adds (around it)
  Mission · Context / Closure        durable objective + closure-aware context
  Authority · Replay                 approval, exactly-once, event-sourced replay
  Verification · Governance          evidence support · freshness · deny-by-default
```

The Pydantic model stays the local application type; ReDevOps emits canonical contracts only at the Mission
boundary and never forces runtime types into the Pydantic node.

## Install

One Python env (`redevops-integrations/.venv`): the Mission SDK, Closure Resolution, `nvidia-nat`, and `pydantic-ai`
coexist — `pydantic-ai-slim[openai]` installs clean (brings `pydantic-graph`; torch/numpy/pydantic untouched).

```bash
VIRTUAL_ENV=.venv uv pip install "pydantic-ai-slim[openai]"   # already present in redevops-integrations/.venv
```

## Run

```bash
cd integrations/pydanticai
../../.venv/bin/python conformance.py                  # PA-A..E + classification tally
../../.venv/bin/python examples/governed_strategy.py   # PA-A/PA-C: typed intent → governed action + typed replay
../../.venv/bin/python examples/typed_verification.py  # PA-D: schema validity vs runtime admissibility (no LLM)
../../.venv/bin/python examples/closure_verdict.py     # PA-E: typed verdict, naive vs closure (gpt-4o-mini, cached)
```

## Expected output (the measured proof)

```
PA-A/PA-C      0 unauthorized · 0 side-effects before approval · exactly 1 after · 0 duplicates on typed replay
               (real pydantic-ai intent: rebalance TLT; typed-state hash reconstructs identically)
PA-D           4 schema-valid outputs the runtime refuses to act on (unsupported · stale · no-authority · governed-deny)
               1 admissible passes both layers — Pydantic valid ≠ ReDevOps admissible
PA-E SciFact   verification accuracy naive 0.800 → closure 0.867 · evidence recall 0.841 → 0.877
               selective autonomy 0.889 @ 90% coverage
PA-B telemetry mission → agent → model (Mission is the causal root)
conformance    12 REDEVOPS_DELTA · 1 PARITY · 1 EXPECTED_IMPLEMENTATION_DIFFERENCE · 0 BUG
```

**Honesty line:** PA-D is deterministic and decisive — the two layers are provably orthogonal. PA-E's closure gain
is modest because SciFact is content-saturated (+0.067, the same closure mechanism NVIDIA's NV-A measures on a
different framework agent — that cross-framework agreement is the semantic-conformance point, not a ranking).

## Acceptance bundle (reproducibility)

Each run writes a reproducibility record to `results/` — framework version, model ids, dataset hashes, per-finding
classification, and a `result_digest`:

```
results/pa_a_governed_strategy.json   results/pa_d_typed_verification.json   results/pa_e_closure_verdict.json
```

## Layout

```
integrations/
  common/                 # provider-neutral harness (shared with nvidia/ and every later integration)
  pydanticai/
    adapters.py              # PydanticAgentCapability (wraps a real pydantic-ai Agent) + PydanticStateProjection
    examples/                # governed_strategy (PA-A/PA-C) · typed_verification (PA-D) · closure_verdict (PA-E)
    conformance.py           # runs PA-A..E, prints the classification tally
    results/                 # emitted acceptance bundles
```

Same `common/` adapter contract as the NVIDIA slice — new framework support is a conformance exercise, not a new
architecture.
