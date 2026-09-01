# AWS (Strands Agents) × ReDevOps — reference implementation

> **AWS Strands / Bedrock own the agent loop and IAM identity. ReDevOps composes Mission authority with IAM as
> deny-wins, runs a low-risk shadow, and adds approval and replay.**
>
> A real `strands-agents` **1.54.0** agent runs **locally, in-process** with the Mission SDK — an OpenAI-compatible
> model provider. **No AWS credentials.** No mock.

This follows the reference template: **reference implementation → benchmark evidence → architecture → reproducible
example.** It is framed as *conformance + incremental runtime value*, never a "Strands vs ReDevOps" scorecard.

**The differentiation for AWS is precise.** AWS's authority model is IAM, whose defining rule is that an explicit
Deny overrides any Allow. So we test the boundary immediately outside a Strands agent's tool calls, on the Wealth
Manager workload. The agent proposes portfolio actions across household accounts; ReDevOps composes three policy
layers with AWS semantics — an organizational guardrail (SCP), the role's IAM policy, and the Mission grant — and
admits an action only when IAM allows it, no explicit Deny fires, and the Mission grants it. And it runs the whole
thing in **shadow** first: propose across 500 households with zero real side effects, measure what the policy would
admit, and promote to execution only through a governed Mission.

## Architecture

```
AWS Strands / Bedrock               what AWS owns
  Agent · model provider · tools      the agent loop
  IAM identity + policy               authentication and the base policy engine
        │
        ▼
ReDevOps Runtime                   what ReDevOps adds (around it)
  IAM-composed authority             SCP ⊕ IAM ⊕ Mission, evaluated deny-wins
  Shadow deployment                  propose across 500 households, 0 side effects, measure first
  Approval · Replay                  exactly-once promoted execution, event-sourced replay
```

## Install

Runs locally, no AWS cloud. Coexists in the shared benchmark venv — installing `strands-agents` added three
packages and left pydantic (2.12.5), numpy and torch untouched.

```bash
VIRTUAL_ENV=.venv uv pip install strands-agents      # already present in redevops-integrations/.venv
```

## Run

```bash
cd integrations/aws
../../.venv/bin/python conformance.py                    # AW-A..E + classification tally
../../.venv/bin/python examples/deny_wins_iam.py         # AW-A: composed policy, deny-wins (real Strands agent)
../../.venv/bin/python examples/shadow_promotion.py      # AW-B: 500-household shadow + governed promotion
../../.venv/bin/python examples/real_wealth_agent.py     # AW-C: a real Strands agent proposes, policy-audited (cached)
```

## Expected output (the measured proof)

```
AW-A deny-wins          native runs 4/4 policy-denied actions · ReDevOps denies all (0 wrong) · 2/2 permitted admitted
AW-B shadow+promote     500 households shadowed with 0 real side effects · promotion 0 unauthorized · exactly-1 · 0 dups
AW-C real agent         4/4 scenarios proposed · the 'withdraw all cash' proposal denied by the SCP guardrail
AW-D explicit deny      a broad Allow on portfolio:* is overridden by an explicit Deny (implicit deny on unlisted)
AW-E telemetry          mission → agent → model (Mission is the causal root)
conformance             8 REDEVOPS_DELTA · 2 PARITY · 0 EXPECTED_IMPLEMENTATION_DIFFERENCE · 0 BUG
```

**Honesty line:** the composed-policy evaluation is deterministic (real IAM/SCP semantics), and the value is the
enforced mechanism. The real Strands agent genuinely proposes the actions — including, in AW-C, a withdrawal the
guardrail then denies before any money moves. IAM identity is modelled as policy documents, not a live AWS call.

## Acceptance bundle (reproducibility)

```
results/aw_a_deny_wins_iam.json   results/aw_b_shadow_promotion.json   results/aw_c_real_wealth_agent.json
```

## Layout

```
integrations/
  common/                 # provider-neutral harness (shared with the framework + Azure + Google slices)
  aws/
    adapters.py              # StrandsAgentCapability (real Strands agent) · Policy/Statement (IAM eval) · ComposedAuthority
    examples/                # deny_wins_iam (AW-A) · shadow_promotion (AW-B) · real_wealth_agent (AW-C)
    conformance.py           # runs AW-A..E, prints the classification tally
    results/                 # emitted acceptance bundles
```

Same `common/` adapter contract as every other slice. Runs in the shared venv (Strands did not force a
dependency-floor change).
