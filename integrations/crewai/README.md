# CrewAI × ReDevOps — reference implementation

> **CrewAI owns role agents, crews and delegation. ReDevOps supplies the multi-agent runtime properties that
> delegation alone doesn't establish — starting with an authority envelope that narrows across every handoff.**
>
> Real `crewai` **1.15.18** Agents, Crews and Tools run **in-process** with the Mission SDK and governance. No mock.

This follows the reference template: **reference implementation → benchmark evidence → architecture → reproducible
example.** It is framed as *conformance + incremental runtime value*, never a "CrewAI vs ReDevOps" scorecard.

**The differentiation for CrewAI is precise.** CrewAI's strongest abstraction is multi-agent delegation: a manager
hands work to coworkers, who run with their own tools. So we test the boundary *immediately outside* delegation.
In CrewAI there is no notion that a delegate may not exceed the delegator's authority — the coworker's tools simply
fire. ReDevOps makes a delegated agent's effective grants the **intersection** of the delegation chain (authority
is monotone non-increasing) and admits a consequential tool call only if its required permission is inside that
narrowed envelope. It adds approval, exactly-once execution and replay around the crew's consequential actions.

## Architecture

```
CrewAI                              what CrewAI owns
  role Agents · Crews · Tasks         the agents, the crew, the delegation tool
  hierarchical / sequential process   multi-agent orchestration
        │
        ▼
ReDevOps Runtime                   what ReDevOps adds (around it)
  Authority envelope                 effective grants = intersection of the delegation chain (never widens)
  Approval · Replay                  exactly-once side effects, event-sourced replay
  Governance · Verification          deny-by-default, no authority pooling, evidence-grounded decisions
```

## Install

One Python env (`redevops-integrations/.venv`): the Mission SDK, Closure Resolution, `nvidia-nat`, `pydantic-ai`,
`llama-index-core` and `crewai` coexist — installing crewai pinned `pydantic` to 2.12.5 and `protobuf` to 6.x
(torch/numpy untouched); the NVIDIA, PydanticAI and LlamaIndex suites reproduce their exact numbers at that floor.

```bash
VIRTUAL_ENV=.venv uv pip install crewai      # already present in redevops-integrations/.venv
```

## Run

```bash
cd integrations/crewai
../../.venv/bin/python conformance.py                    # CR-A..E + classification tally
../../.venv/bin/python examples/delegation_authority.py  # CR-A: a delegate cannot widen authority (no LLM)
../../.venv/bin/python examples/governed_delegation.py   # CR-B: crew recommends, Mission governs the action
../../.venv/bin/python examples/crew_decision.py         # CR-C: a real 2-agent crew triages AML alerts (cached)
```

## Expected output (the measured proof)

```
CR-A delegation authority   native runs 3/3 authority-widening actions · ReDevOps denies all 3 (0 wrong)
                            in-scope delegated actions admitted 2/2 (incl. a legitimately-held freeze)
CR-B governed action        0 unauthorized · 0 before approval · exactly 1 after · 0 duplicates on replay
CR-C real crew (n=6)        crew decision accuracy 1.000, wrapped as a Mission (ReDevOps preserves the decision)
CR-D no authority pooling   a crew of low-authority agents cannot sum to a grant none of them hold
CR-E telemetry              mission → crew → agent → model (Mission is the causal root)
conformance                 7 REDEVOPS_DELTA · 2 PARITY · 1 EXPECTED_IMPLEMENTATION_DIFFERENCE · 0 BUG
```

**Honesty line:** CR-A is deterministic and decisive — authority narrowing is a mechanism, not a model behaviour.
CR-C's crew is accurate on a small, clean alert set, so the evidence-grounding gate routes nothing here; it is the
*same* gate the LlamaIndex slice's LI-B shows biting hard (6/6 unsupported answers abstained). The value is the
enforced gate, reported straight — not a headline number.

## Acceptance bundle (reproducibility)

Each run writes a reproducibility record to `results/` — framework version, model ids, per-finding classification,
and a `result_digest`:

```
results/cr_a_delegation_authority.json   results/cr_b_governed_delegation.json   results/cr_c_crew_decision.json
```

## Layout

```
integrations/
  common/                 # provider-neutral harness (shared with nvidia/, pydanticai/, llamaindex/)
  crewai/
    adapters.py              # AuthorityEnvelope (narrows across delegation) · RedevopsGovernedTool (real crewai tool)
                             #   · CrewCapability (wraps a real crewai Crew)
    examples/                # delegation_authority (CR-A) · governed_delegation (CR-B) · crew_decision (CR-C)
    conformance.py           # runs CR-A..E, prints the classification tally
    results/                 # emitted acceptance bundles
```

Same `common/` adapter contract as the NVIDIA, PydanticAI and LlamaIndex slices — new framework support is a
conformance exercise, not a new architecture.
