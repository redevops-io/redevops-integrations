# LlamaIndex × ReDevOps — reference implementation

> **LlamaIndex owns indexing, retrieval and the query engine. ReDevOps composes the retriever with structural
> closure and supplies the runtime properties retrieval alone doesn't establish.**
>
> A real `llama-index-core` **0.14.24** `VectorStoreIndex` retriever and query engine run **in-process** with the
> Mission SDK, Closure Resolution and governance. Embeddings are local (bge-small, offline CPU). No mock.

This follows the reference template: **reference implementation → benchmark evidence → architecture → reproducible
example.** It is framed as *conformance + incremental runtime value*, never a "LlamaIndex vs ReDevOps" scorecard.

**The differentiation for LlamaIndex is precise.** LlamaIndex's strongest abstraction is retrieval over an index,
so we test the boundary *immediately outside* it. A semantic retriever finds what is textually similar to a query;
it does not, by construction, recover a code change's **contract callers** and transitive callers (structurally
linked, not similar), and its query engine synthesizes a confident answer **whether or not** retrieval found the
needle. ReDevOps registers the LlamaIndex retriever as **one representation** and composes it with the static
call-graph closure, and adds evidence-support verification, authority, governance and replay around the answer.
The retriever is composed and preferred when it suffices — never replaced.

## Architecture

```
LlamaIndex                          what LlamaIndex owns
  VectorStoreIndex · retriever        indexing, embeddings, retrieval
  query engine · Workflows            response synthesis + local control flow
        │
        ▼
ReDevOps Runtime                   what ReDevOps adds (around it)
  Context / Closure                  the retriever as one representation, composed with structural closure
  Authority · Replay                 approval, exactly-once, event-sourced replay
  Verification · Governance          evidence support · abstain-on-unsupported · deny-by-default
```

## Install

One Python env (`redevops-integrations/.venv`): the Mission SDK, Closure Resolution, `nvidia-nat`, `pydantic-ai` and
`llama-index-core` coexist — installing llama-index-core only added packages (torch/numpy/pydantic untouched).

```bash
VIRTUAL_ENV=.venv uv pip install llama-index-core     # already present in redevops-integrations/.venv
```

## Run

```bash
cd integrations/llamaindex
../../.venv/bin/python conformance.py                     # LI-A..E + classification tally
../../.venv/bin/python examples/code_closure.py           # LI-A: code closure just outside retrieval (no LLM)
../../.venv/bin/python examples/longcontext_answer.py     # LI-B: query-engine synthesis vs verified abstention (cached)
../../.venv/bin/python examples/governed_retrieval.py     # LI-D: retrieval informs, ReDevOps governs the action
```

## Expected output (the measured proof)

```
LI-A code closure   closure recall 0.320 → 0.693 · contract-caller recall 0.300 → 0.600 (1138 agentic-os symbols)
                    0 regressions; on tasks retrieval already resolves, composition preserves it
LI-B long-context   query-engine accuracy: needle retrieved 1.000 · needle missed 0.000
                    on a miss 4/6 confidently answered → ReDevOps abstains 6/6 · selective accuracy 1.000 @ 50%
LI-D governed       0 unauthorized · 0 before approval · exactly 1 after · 0 duplicates on replay
LI-E telemetry      mission → retriever → embed (Mission is the causal root)
conformance         10 REDEVOPS_DELTA · 1 PARITY · 1 EXPECTED_IMPLEMENTATION_DIFFERENCE · 0 BUG
```

**Honesty line:** LI-A's flagship result is real and modest — semantic retrieval recovers the seed and type edges
but misses roughly half the structural closure; composing it with the call graph roughly doubles both closure and
contract-caller recall while never degrading a task LlamaIndex already resolved. The composed arm is structurally
seeded (a different input than NL retrieval), so its match recall is high by construction and is **not** claimed as
a retrieval win — the meaningful deltas are closure and contract-caller recall.

## Acceptance bundle (reproducibility)

Each run writes a reproducibility record to `results/` — framework version, model ids, dataset hashes, per-finding
classification, and a `result_digest`:

```
results/li_a_code_closure.json   results/li_b_longcontext_answer.json   results/li_d_governed_retrieval.json
```

## Layout

```
integrations/
  common/                 # provider-neutral harness (shared with nvidia/, pydanticai/ and every later integration)
  llamaindex/
    adapters.py              # LlamaIndexRetrieverCapability (real VectorStoreIndex) + LocalEmbedding (offline bge-small)
    examples/                # code_closure (LI-A) · longcontext_answer (LI-B) · governed_retrieval (LI-D)
    conformance.py           # runs LI-A..E, prints the classification tally
    results/                 # emitted acceptance bundles
```

Same `common/` adapter contract as the NVIDIA and PydanticAI slices — new framework support is a conformance
exercise, not a new architecture.
