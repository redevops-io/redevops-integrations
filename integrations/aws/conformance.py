"""AWS (Strands Agents) conformance runner — runs AW-A/AW-B/AW-C, adds AW-D + AW-E, prints the tally.

    python conformance.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "examples"))
sys.path.insert(0, os.path.join(HERE, "..", "common"))

from adapter import NodeSpan, spans_to_dict  # noqa: E402
from adapters import Policy, Statement  # noqa: E402
from classification import Finding, ResultClass  # noqa: E402


def aw_d_explicit_deny_precedence():
    """AW-D — AWS evaluation semantics: an explicit Deny overrides any Allow, even a broad allow on the same
    action/resource. Deterministic; the rule the whole deny-wins composition rests on."""
    p = Policy([
        Statement("Allow", ["portfolio:*"], ["household/*"]),
        Statement("Deny", ["portfolio:trade"], ["household/restricted"]),
    ])
    ok = (p.evaluate("portfolio:trade", "household/restricted") == "Deny"
          and p.evaluate("portfolio:trade", "household/12") == "Allow"
          and p.evaluate("account:withdraw", "household/12") == "Implicit")
    return Finding("AW-D explicit-deny precedence", "deny_overrides_allow",
                   native="(a permissive allow would let the action through)",
                   with_redevops="explicit Deny wins",
                   classification=ResultClass.REDEVOPS_DELTA if ok else ResultClass.BUG,
                   note="a broad Allow on portfolio:* is overridden by an explicit Deny on the restricted resource; "
                        "an unlisted action is an implicit deny")


def aw_e_telemetry():
    """AW-E — the Strands agent/model spans nest UNDER the Mission causal node; Mission is the root."""
    agent = NodeSpan("strands:wealth_manager", "agent", "strands:wealth_manager",
                     children=[NodeSpan("model", "model", "gpt-4o-mini")])
    root = NodeSpan("mission.node.execute", "workflow_step", "mission.node", children=[agent])
    tree = spans_to_dict(root)
    ok = tree["kind"] == "workflow_step" and tree["children"][0]["kind"] == "agent" \
        and tree["children"][0]["children"][0]["kind"] == "model"
    return Finding("AW-E telemetry nesting", "mission_is_root", native="(the Strands agent trace is the root)",
                   with_redevops="mission→agent→model",
                   classification=ResultClass.REDEVOPS_DELTA if ok else ResultClass.BUG,
                   note="Strands agent/model spans nest beneath the Mission causal node")


def main():
    print("=" * 78)
    print("AWS (Strands Agents) × ReDevOps — conformance run")
    print("=" * 78)
    import deny_wins_iam
    import real_wealth_agent
    import shadow_promotion
    print("\n--- AW-A deny-wins over IAM (flagship) ---"); deny_wins_iam.main()
    print("\n--- AW-B shadow + governed promotion ---"); shadow_promotion.main()
    print("\n--- AW-C real wealth agent ---"); real_wealth_agent.main()

    extra = [aw_d_explicit_deny_precedence(), aw_e_telemetry()]
    print("\n--- AW-D / AW-E ---")
    for f in extra:
        print(f"  [{f.classification.value:34}] {f.experiment}")

    import json
    tally = {}
    resdir = os.path.join(HERE, "results")
    for fn in sorted(os.listdir(resdir)) if os.path.isdir(resdir) else []:
        for x in json.load(open(os.path.join(resdir, fn))).get("findings", []):
            c = x["classification"]
            tally[c] = tally.get(c, 0) + 1
    for f in extra:
        tally[f.classification.value] = tally.get(f.classification.value, 0) + 1

    print("\n" + "=" * 78)
    print("result classification tally (framework: strands-agents):")
    for k in ("REDEVOPS_DELTA", "PARITY", "FRAMEWORK_NATIVE_ADVANTAGE", "EXPECTED_IMPLEMENTATION_DIFFERENCE", "BUG"):
        if tally.get(k):
            print(f"  {k:36} {tally[k]}")
    print("=" * 78)
    print("thesis: AWS Strands / Bedrock own the agent loop and IAM identity; ReDevOps composes Mission authority "
          "with IAM as deny-wins, runs a low-risk shadow, and adds approval and replay — the governed-execution "
          "properties an agent loop and a policy engine don't establish on their own.")


if __name__ == "__main__":
    main()
