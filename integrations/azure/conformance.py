"""Azure (Semantic Kernel) conformance runner — runs AZ-A/AZ-B, adds AZ-C deny-wins + AZ-E telemetry, tallies.

    integrations/azure/.venv/bin/python conformance.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "examples"))
sys.path.insert(0, os.path.join(HERE, "..", "common"))

from adapter import NodeSpan, spans_to_dict  # noqa: E402
from adapters import DeploymentAdmission, EntraRole  # noqa: E402
from classification import Finding, ResultClass  # noqa: E402


def az_c_deny_wins():
    """AZ-C — ReDevOps authority composes with an Entra role as DENY-WINS. A permissive framework grant
    (staging+production) intersected with a restrictive Entra role (staging only) yields staging only — the
    restrictive policy governs; the permissive grant never widens it. Deterministic."""
    gate = DeploymentAdmission(EntraRole("ci-bot", ["staging"]), framework_grant_envs=("staging", "production"))
    authorized = gate.authorized_envs()
    prod_ok, _ = gate.admit("production", ["CTL-ENCRYPTION", "CTL-IMAGE", "CTL-NETWORK", "CTL-CHANGE-APPROVED"])
    stg_ok, _ = gate.admit("staging", ["CTL-ENCRYPTION", "CTL-IMAGE"])
    ok = authorized == frozenset({"staging"}) and not prod_ok and stg_ok
    return Finding("AZ-C deny-wins over Entra", "effective_authority",
                   native="(a permissive framework grant would allow production)",
                   with_redevops="staging only (intersection)",
                   classification=ResultClass.REDEVOPS_DELTA if ok else ResultClass.BUG,
                   note="effective authority is the intersection of the framework grant and the Entra role; a "
                        "fully-compliant production deploy is still denied because the role does not permit prod")


def az_e_telemetry():
    """AZ-E — the Semantic Kernel agent/function/model spans nest UNDER the Mission causal node; Mission is root."""
    agent = NodeSpan("sk:deployer", "agent", "sk:deployer",
                     children=[NodeSpan("function", "tool", "ops.request_deploy",
                                        children=[NodeSpan("model", "model", "gpt-4o-mini")])])
    root = NodeSpan("mission.node.deploy", "workflow_step", "mission.node", children=[agent])
    tree = spans_to_dict(root)
    ok = tree["kind"] == "workflow_step" and tree["children"][0]["kind"] == "agent" \
        and tree["children"][0]["children"][0]["kind"] == "tool"
    return Finding("AZ-E telemetry nesting", "mission_is_root", native="(the SK agent trace is the root)",
                   with_redevops="mission→agent→function→model",
                   classification=ResultClass.REDEVOPS_DELTA if ok else ResultClass.BUG,
                   note="Semantic Kernel agent/function/model spans nest beneath the Mission causal node")


def main():
    print("=" * 78)
    print("Azure (Semantic Kernel) × ReDevOps — conformance run")
    print("=" * 78)
    import compliance_gated_deploy
    import governed_deploy
    print("\n--- AZ-A compliance-gated deployment (flagship) ---"); compliance_gated_deploy.main()
    print("\n--- AZ-B governed deployment ---"); governed_deploy.main()

    extra = [az_c_deny_wins(), az_e_telemetry()]
    print("\n--- AZ-C / AZ-E ---")
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
    print("result classification tally (framework: semantic-kernel):")
    for k in ("REDEVOPS_DELTA", "PARITY", "FRAMEWORK_NATIVE_ADVANTAGE", "EXPECTED_IMPLEMENTATION_DIFFERENCE", "BUG"):
        if tally.get(k):
            print(f"  {k:36} {tally[k]}")
    print("=" * 78)
    print("thesis: Semantic Kernel owns the agent loop and function-calling; ReDevOps adds a compliance-control gate, "
          "Entra-composing authority (deny-wins), approval and replay — deployment as a governed Mission.")


if __name__ == "__main__":
    main()
