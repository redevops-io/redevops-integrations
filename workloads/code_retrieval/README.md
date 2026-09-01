# Code-retrieval benchmark — Phase 0

Does the Context Runtime preserve the **dependency closure** a code change requires — not just the nearest
snippet? Full plan + rationale: [`docs/plans/code_dependency_retrieval_test_plan.md`](../../docs/plans/code_dependency_retrieval_test_plan.md).

**Discipline:** the plugin is an experimentally *earned* feature. Phase 0 tests the **existing CR only**;
the schema and oracle are **frozen before any arm runs**; the decision gate decides whether Phase 1 (a
retrieval capability) happens at all — and whether it's broad graph expansion or a narrow *contract-impact*
representation (H3b: *minimum sufficient change closure*, not maximum dependency closure).

## What's here (build order, steps 1–2 done)

- **`frozen_spec.yaml`** — the committed pre-registration: hypotheses (H1/H3/H3b), the 7 dependency classes,
  the resolution definition, the equal-budget requirement, the arms, the metrics (incl. **closure
  precision** and the headline **resolution-efficiency = resolution per 1K materialized tokens subject to
  required-edge recall**), the pre-registered regression, and the 4-row decision gate. *Frozen: do not edit
  definitions once a result exists — bump `spec_version` and re-freeze.*
- **`oracle/`** — the closure oracle, declared **independently of any patch**:
  - `oracle/tasks/*.yaml` — a task declares its **seed symbols + required dependency classes** (symbols,
    **not filenames**); symbols resolve to files only at scoring time. Empty class = an explicit claim the
    change does *not* require it.
  - `oracle/hash_declaration.py` — canonicalizes a declaration (minus the hash) and stamps a sha256, so the
    target is provably fixed **before** the reference patch is written.
  - Two families declared + frozen so far: `cap_metadata_propagation` (capability-metadata propagation) and
    `signature_change` (interface/signature change — callers in a different file, no test/config edges).
- **`scoring.py` + `test_scoring.py`** *(step 3, done)* — the arm-agnostic scorer: an arm reports the
  **symbols it surfaced** (each with file · line range · token cost) and the scorer emits every
  `frozen_spec` metric — per-class recall, total closure recall, contract-caller recall, **closure
  precision**, materialized-token waste, context efficiency. Tests prove the **H1 shape** (a naive arm gets
  `match_recall=1.0` but `contract_caller_recall=0.0`, `closure_recall=0.5`) and that over-retrieval is
  penalized by precision/waste, not recall. `resolution_efficiency` is filled at run time from the pass/fail
  outcome + the materialized-token denominator the scorer emits.

## Runnable harness (steps 4–5, model-free arms DONE)

- **`resolver.py`** — Python-AST symbol index (`module:symbol` → file · line range · token cost). Shared by
  every arm to *report* what it surfaced; it does **no** graph expansion (it is not the plugin). Verified: all
  required oracle symbols resolve against the pinned tree (`f8ce79c`, 0 missing).
- **`retrievers.py`** — two model-free arms that need no external stack: `oracle_closure` (the ceiling — return
  the declared closure) and `naive_semantic` (self-contained BM25 over symbol sources; the honest low baseline).
- **`run.py`** — indexes the repo, runs the arms at a matched token budget, scores every family, writes
  `results_phase0.json`. `existing_cr` / `existing_cr + oracle-hints` print **PENDING** (need the live CR stack)
  rather than a faked number.

- **`apply_loop.py`** *(step 6, P4)* — the resolution loop: per arm, isolate a git worktree of
  `agentic-os@f8ce79c`, give the agent only that arm's context, apply its edits (**anchored** whole-symbol
  splice, re-indexing the worktree each round; or `search_replace`), drop in the frozen acceptance test, run
  pytest. **Localize-verify** (`--verify-rounds`): on failure, feed the failing pytest output back and let the
  agent edit again — a real coding-agent loop that keeps the retrieval signal (a symbol the arm never
  retrieved stays unfixable). Agent = **Qwen3.8-27B** (`:31111`, thinking disabled), pass@2. Frozen acceptance
  tests in `oracle/acceptance/`.

**Retrieval numbers:** [`docs/results/code_retrieval_phase0.md`](../../docs/results/code_retrieval_phase0.md).
Dense hybrid (`existing_cr_hybrid`, bge-small + BM25) **materially beats lexical** — on `cap_metadata` it lifts
closure 0.40→0.80 and contract 0.50→1.00; only `transitive_caller` resists. **Resolution numbers:**
[`docs/results/code_retrieval_resolution.md`](../../docs/results/code_retrieval_resolution.md) — two families
(ambiguous + **compatibility-explicit**) × two agent models (local **Qwen3.8-27B**, cloud **Kimi-k2.7-code**
via `--provider kimi`). Headline: on the compat-explicit family **`oracle_closure` RESOLVES 2/2 on both
models** while partial-context arms fail 0/2 (they miss the seed def) — so **closure recall now determines
resolution** (the H3 direction). On the ambiguous family both models fall into the *identical* required-param
trap → the earlier 0/2 was a task-spec artifact, not a model/retrieval limit. The stronger cloud model changed
no outcome.

**Phase 1 — graph traversal (built + measured):**
[`docs/results/code_retrieval_graph.md`](../../docs/results/code_retrieval_graph.md). A call-graph traversal
arm (`graph_closure`, `resolver.build_call_graph`) and its **union with content retrieval**
(`graph_plus_hybrid`). Graph recovers the **transitive-caller** class content retrieval misses (transitive
family closure `0.50→1.00`) but has its own blind spots (class-instantiation, non-self-receiver calls); the
**union dominates** — closure 1.00 on 3/4 families — and, decisively, **converts to resolution**: on
`signature_change_compat` hybrid 0/2 + graph 0/2 but **union RESOLVED 2/2** (= oracle); on `transitive` **graph
alone RESOLVED 2/2**. So the fix is graph **∪** content (complementary), not graph instead of it.

## Still to build (gated, in order)

- **More families across the miss classes** — type-seeded / instantiation-edge changes (the graph's blind
  spot) and deep transitive chains (content's blind spot) — enough to fit `resolution ~ {match, closure,
  contract-caller}` and to decide whether to extend the graph with instantiation/attribute edges.
- **Bring dense embeds up** (`:8012/:8013`) to test CR's non-lexical retrieval directly (H2).
- **Wire `graph_plus_hybrid` as a real Context Runtime retrieval representation** (a plugin alongside
  bm25/dense/graph) — gated on the above; retrieve *less, structurally*, never broad graph expansion (H3b).
