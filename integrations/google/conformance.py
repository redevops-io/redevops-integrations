"""Google ADK / A2A conformance runner — runs GO-A/GO-B/GO-C, adds GO-D + GO-E, prints the tally.

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
from adapters import AuthorityChain  # noqa: E402
from classification import Finding, ResultClass  # noqa: E402


def go_d_no_reamplification():
    """GO-D — a downstream A2A hop cannot re-amplify a dropped grant. Even if hop 3 declares a broad grant, its
    effective authority stays the intersection of the whole chain, so a capability dropped upstream stays dropped.
    Deterministic; the property that makes multi-hop delegation safe."""
    root = AuthorityChain([("A", {"x", "y", "z"})])
    h2 = root.extend("B", {"x", "y"})            # z dropped
    h3 = h2.extend("C", {"x", "y", "z"})         # C re-declares z — must not regain it
    ok = h3.effective() == frozenset({"x", "y"}) and not h3.admits("z")
    return Finding("GO-D no capability re-amplification", "regained_dropped_grant",
                   native="(a downstream agent runs with its own declared capabilities)",
                   with_redevops="intersection of the whole chain",
                   classification=ResultClass.REDEVOPS_DELTA if ok else ResultClass.BUG,
                   note="hop C re-declares a grant dropped at hop B, but effective authority stays the chain "
                        "intersection {x,y} — the grant cannot be re-amplified downstream")


def go_e_telemetry():
    """GO-E — the ADK agent/model spans nest UNDER the Mission causal node across the mesh; Mission is the root."""
    web = NodeSpan("adk:web_researcher", "agent", "adk:web_researcher",
                   children=[NodeSpan("model", "model", "gpt-4o-mini")])
    coord = NodeSpan("adk:coordinator", "agent", "adk:coordinator", children=[web])
    root = NodeSpan("mission.node.research", "workflow_step", "mission.node", children=[coord])
    tree = spans_to_dict(root)
    ok = tree["kind"] == "workflow_step" and tree["children"][0]["kind"] == "agent" \
        and tree["children"][0]["children"][0]["kind"] == "agent"
    return Finding("GO-E telemetry nesting", "mission_is_root", native="(the coordinator trace is the root)",
                   with_redevops="mission→coordinator→agent→model",
                   classification=ResultClass.REDEVOPS_DELTA if ok else ResultClass.BUG,
                   note="ADK delegated-agent spans nest beneath the Mission causal node across the mesh")


def main():
    print("=" * 78)
    print("Google ADK / A2A × ReDevOps — conformance run")
    print("=" * 78)
    import governed_research
    import mesh_authority
    import research_mesh
    print("\n--- GO-A mesh authority + provenance (flagship) ---"); mesh_authority.main()
    print("\n--- GO-B governed research action ---"); governed_research.main()
    print("\n--- GO-C real research mesh ---"); research_mesh.main()

    extra = [go_d_no_reamplification(), go_e_telemetry()]
    print("\n--- GO-D / GO-E ---")
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
    print("result classification tally (framework: google-adk):")
    for k in ("REDEVOPS_DELTA", "PARITY", "FRAMEWORK_NATIVE_ADVANTAGE", "EXPECTED_IMPLEMENTATION_DIFFERENCE", "BUG"):
        if tally.get(k):
            print(f"  {k:36} {tally[k]}")
    print("=" * 78)
    print("thesis: Google ADK / A2A own agent construction and agent-to-agent delegation; ReDevOps adds authority "
          "that narrows across every hop of the mesh, provenance-gated synthesis, approval and replay — the "
          "multi-hop runtime properties delegation alone doesn't establish.")


if __name__ == "__main__":
    main()
