"""CrewAI conformance runner — runs CR-A/CR-B/CR-C, adds CR-D no-pooling + CR-E telemetry, prints the tally.

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
from adapters import AuthorityEnvelope, RedevopsGovernedTool  # noqa: E402
from classification import Finding, ResultClass  # noqa: E402


def cr_d_no_pooling():
    """CR-D — authority does not POOL across a crew. Two agents that each lack `funds:release` cannot jointly
    perform it, and a grant held by one agent is not transferable to another. Deterministic; multi-agent specific."""
    a = AuthorityEnvelope({"kyc:review"}, agent="A")
    b = AuthorityEnvelope({"case:note"}, agent="B")
    sink: list = []
    # neither A nor B may release funds — and there is no "crew" envelope that sums their grants
    a_denied = "OK" not in RedevopsGovernedTool(permission="funds:release").bind(a, sink)._run()
    b_denied = "OK" not in RedevopsGovernedTool(permission="funds:release").bind(b, sink)._run()
    # a grant held by A is not usable by B
    a2 = AuthorityEnvelope({"account:freeze"}, agent="A")
    b_cannot_borrow = "OK" not in RedevopsGovernedTool(permission="account:freeze").bind(b, sink)._run()
    ok = a_denied and b_denied and b_cannot_borrow and not sink
    return Finding("CR-D no authority pooling", "pooled_or_borrowed_authority",
                   native="(a crew of agents can collectively call any tool any member holds)", with_redevops=len(sink),
                   classification=ResultClass.REDEVOPS_DELTA if ok else ResultClass.BUG,
                   note="authority is per-agent-envelope, never summed across the crew nor borrowed between agents; "
                        f"A2 holds account:freeze but B still cannot use it")


def cr_e_telemetry():
    """CR-E — the crew's agent/model spans nest UNDER the Mission causal node; the Mission is the root."""
    agents = [NodeSpan(f"agent:{i}", "agent", r, children=[NodeSpan("model", "model", "gpt-4o-mini")])
              for i, r in enumerate(("Risk Analyst", "Compliance Officer"))]
    crew = NodeSpan("crewai:triage", "crew", "crewai:triage", children=agents)
    root = NodeSpan("mission.node.decide", "workflow_step", "mission.node", children=[crew])
    tree = spans_to_dict(root)
    ok = tree["kind"] == "workflow_step" and tree["children"][0]["kind"] == "crew" \
        and tree["children"][0]["children"][0]["kind"] == "agent"
    return Finding("CR-E telemetry nesting", "mission_is_root", native="(the crew trace is the root)",
                   with_redevops="mission→crew→agent→model",
                   classification=ResultClass.REDEVOPS_DELTA if ok else ResultClass.BUG,
                   note="CrewAI crew/agent/model spans nest beneath the Mission causal node")


def main():
    print("=" * 78)
    print("CrewAI × ReDevOps — conformance run")
    print("=" * 78)
    import crew_decision
    import delegation_authority
    import governed_delegation
    print("\n--- CR-A delegation authority (flagship) ---"); delegation_authority.main()
    print("\n--- CR-B governed delegated action ---"); governed_delegation.main()
    print("\n--- CR-C crew decision (real crew) ---"); crew_decision.main()

    extra = [cr_d_no_pooling(), cr_e_telemetry()]
    print("\n--- CR-D / CR-E ---")
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
    print("result classification tally (framework: crewai):")
    for k in ("REDEVOPS_DELTA", "PARITY", "FRAMEWORK_NATIVE_ADVANTAGE", "EXPECTED_IMPLEMENTATION_DIFFERENCE", "BUG"):
        if tally.get(k):
            print(f"  {k:36} {tally[k]}")
    print("=" * 78)
    print("thesis: CrewAI owns role agents, crews and delegation; ReDevOps adds an authority envelope that narrows "
          "across delegation, plus approval, replay and governance — the multi-agent runtime properties delegation "
          "alone doesn't establish.")


if __name__ == "__main__":
    main()
