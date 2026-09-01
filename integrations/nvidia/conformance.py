"""NVIDIA conformance runner — runs NV-A..E, adds the telemetry/optimization checks, prints the result table.

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
from adapters import NemoAgentCapability, NemoTelemetryAdapter  # noqa: E402
from classification import Finding, ResultClass  # noqa: E402


def nv_e_telemetry():
    """NV-E — native NeMo spans nest UNDER the Mission node; the Mission is the causal root, never a substrate span."""
    agent = NodeSpan("nat:agent:x", "agent", "nat:scifact_verifier",
                     children=[NodeSpan("model", "model", "gpt-4o-mini")])
    root = NemoTelemetryAdapter.nest("mission.node.verify", agent)
    tree = spans_to_dict(root)
    ok = tree["kind"] == "workflow_step" and tree["children"][0]["kind"] == "agent" \
        and tree["children"][0]["children"][0]["kind"] == "model"
    return Finding("NV-E telemetry nesting", "mission_is_root", native="(agent trace is the root)",
                   with_redevops="mission→agent→model", classification=ResultClass.REDEVOPS_DELTA if ok else ResultClass.BUG,
                   note="NeMo model/tool spans nest beneath the Mission causal node")


def nv_c_optimization():
    """NV-C — separation of concerns: NeMo optimizes agent config; ReDevOps optimizes representation/context/plan;
    Mission semantics stay stable. Architectural (not a task-score comparison)."""
    return Finding("NV-C optimization separation", "mission_semantics", native="agent config (NeMo)",
                   with_redevops="context/representation/plan (ReDevOps)",
                   classification=ResultClass.EXPECTED_IMPLEMENTATION_DIFFERENCE,
                   note="two independent optimization layers; the Mission objective/authority/evidence are invariant")


def main():
    print("=" * 78)
    print("NVIDIA NeMo Agent Toolkit × ReDevOps — conformance run")
    print("=" * 78)
    import gdpr_structural
    import governed_tool
    import scifact_closure
    print("\n--- NV-A SciFact ---"); scifact_closure.main()
    print("\n--- NV-B GDPR ---"); gdpr_structural.main()
    print("\n--- NV-D governed tool ---"); governed_tool.main()

    extra = [nv_e_telemetry(), nv_c_optimization()]
    print("\n--- NV-C / NV-E ---")
    for f in extra:
        print(f"  [{f.classification.value:34}] {f.experiment}")

    # aggregate classification table across all emitted result bundles
    import json
    tally = {}
    resdir = os.path.join(HERE, "results")
    for fn in sorted(os.listdir(resdir)) if os.path.isdir(resdir) else []:
        for x in json.load(open(os.path.join(resdir, fn))).get("findings", []):
            c = x["classification"]
            tally[c] = tally.get(c, 0) + 1
    for f in extra:
        tally[f.classification.value] = tally.get(f.classification.value, 0) + 1

    cap = NemoAgentCapability(system="x")
    print("\n" + "=" * 78)
    print(f"result classification tally (framework: {cap.framework} {cap.framework_version}):")
    for k in ("REDEVOPS_DELTA", "PARITY", "FRAMEWORK_NATIVE_ADVANTAGE", "EXPECTED_IMPLEMENTATION_DIFFERENCE", "BUG"):
        if tally.get(k):
            print(f"  {k:36} {tally[k]}")
    print("=" * 78)
    print("thesis: NeMo owns the agent loop; ReDevOps adds closure-aware context + governed Mission semantics.")


if __name__ == "__main__":
    main()
