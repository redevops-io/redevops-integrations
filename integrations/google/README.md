# Google ADK / A2A × ReDevOps — reference implementation

> **Google ADK / A2A own agent construction and agent-to-agent delegation. ReDevOps adds authority that narrows
> across every hop of the mesh, provenance-gated synthesis, approval and replay.**
>
> A real `google-adk` **2.8.0** agent runs **locally, in-process** with the Mission SDK — LiteLlm against any
> OpenAI-compatible endpoint. **No Google Cloud credentials.** No mock.

This follows the reference template: **reference implementation → benchmark evidence → architecture → reproducible
example.** It is framed as *conformance + incremental runtime value*, never a "ADK vs ReDevOps" scorecard.

**The differentiation for Google is precise, and distinct from the single-hop CrewAI slice.** ADK / A2A's strongest
abstraction is agent-to-agent delegation across service boundaries — an agent hands a task to another agent, which
may hand it on again. Two failures are unique to a *multi-hop mesh*: **transitive authority escalation** (a 3rd-hop
agent doing what an intermediate hop was never authorized for) and **loss of provenance** (a coordinator
synthesizing a finding it cannot attribute to an authorized agent). ReDevOps makes effective authority the
intersection of the **whole chain** — monotone non-increasing across every hop, so a grant dropped upstream cannot
be re-amplified downstream — and admits a finding into the synthesis only when its provenance chain is attributable
to authorized agents.

## Architecture

```
Google ADK / A2A                    what ADK / A2A own
  Agent · LiteLlm · Runner            agent construction + the agent loop
  agent-to-agent delegation           the multi-hop mesh
        │
        ▼
ReDevOps Runtime                   what ReDevOps adds (around it)
  Chain-wide authority               effective grants = intersection of every hop (no re-amplification)
  Provenance-gated synthesis         only attributable findings enter the coordinator's answer
  Approval · Replay                  exactly-once consequential actions, event-sourced replay
```

## Install

Runs locally, no Google Cloud. Coexists in the shared benchmark venv — installing `google-adk[extensions]` and
`a2a-sdk` left pydantic (held at 2.12.5), numpy and torch untouched; only websockets moved a patch version.

```bash
VIRTUAL_ENV=.venv uv pip install "google-adk[extensions]" a2a-sdk   # already present in redevops-integrations/.venv
```

## Run

```bash
cd integrations/google
../../.venv/bin/python conformance.py                    # GO-A..E + classification tally
../../.venv/bin/python examples/mesh_authority.py        # GO-A: chain-wide attenuation + provenance (no LLM)
../../.venv/bin/python examples/governed_research.py     # GO-B: a delegated finding, governed as a Mission
../../.venv/bin/python examples/research_mesh.py         # GO-C: a real ADK agent answers delegated sub-questions
```

## Expected output (the measured proof)

```
GO-A mesh authority     transitive: native runs 2/2 escalations · ReDevOps denies all · in-scope 2/2 admitted
                        provenance: native trusts 2/2 unprovenanced findings · ReDevOps excludes all
GO-B governed action    0 unauthorized · 0 before approval · exactly 1 after · 0 duplicates on replay
GO-C real ADK mesh      4/4 sub-questions answered end-to-end · 4/4 findings provenance-attributable
GO-D no re-amplification a grant dropped at an intermediate hop stays dropped (chain intersection)
GO-E telemetry          mission → coordinator → agent → model (Mission is the causal root)
conformance             9 REDEVOPS_DELTA · 2 PARITY · 0 EXPECTED_IMPLEMENTATION_DIFFERENCE · 0 BUG
```

**Honesty line:** GO-A is deterministic — chain-wide attenuation and provenance are mechanisms, not model
behaviours. The real ADK agent genuinely runs the research sub-questions in GO-C; what ReDevOps changes is that
authority narrows across every hop and every admitted finding is attributable.

## Acceptance bundle (reproducibility)

```
results/go_a_mesh_authority.json   results/go_b_governed_research.json   results/go_c_research_mesh.json
```

## Layout

```
integrations/
  common/                 # provider-neutral harness (shared with the framework + Azure slices)
  google/
    adapters.py              # AdkAgentCapability (real ADK+LiteLlm) · AuthorityChain (multi-hop) · Provenance
    examples/                # mesh_authority (GO-A) · governed_research (GO-B) · research_mesh (GO-C)
    conformance.py           # runs GO-A..E, prints the classification tally
    results/                 # emitted acceptance bundles
```

Same `common/` adapter contract as every other slice — new provider support is a conformance exercise, not a new
architecture. Runs in the shared venv (unlike the Azure slice, ADK did not force a dependency-floor change).
