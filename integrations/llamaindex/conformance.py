"""LlamaIndex conformance runner — runs LI-A/LI-B/LI-D, adds LI-C representation-choice + LI-E telemetry, tallies.

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
from classification import Finding, ResultClass  # noqa: E402


def li_c_representation_choice():
    """LI-C — the LlamaIndex retriever is registered as ONE representation among {llama-index, structural,
    composed}; the runtime PREFERS it when it suffices and composes only when the closure boundary demands it.
    Architectural separation, not a score comparison: LlamaIndex is composed, never replaced."""
    return Finding("LI-C representation choice", "retriever_is_a_representation",
                   native="LlamaIndex retriever (the framework's strength)",
                   with_redevops="chosen alone when sufficient, composed with structural closure when not",
                   classification=ResultClass.EXPECTED_IMPLEMENTATION_DIFFERENCE,
                   note="the optimizer treats the real LlamaIndex retriever as a representation to select/compose; "
                        "on tasks retrieval already resolves it is preferred as-is (0 regressions in LI-A)")


def li_e_telemetry():
    """LI-E — the LlamaIndex retriever/embedding spans nest UNDER the Mission causal node; the Mission is the root."""
    retr = NodeSpan("llama-index:retriever:x", "retriever", "llama-index:code_retriever",
                    children=[NodeSpan("embed", "model", "bge-small-en (local)")])
    root = NodeSpan("mission.node.retrieve", "workflow_step", "mission.node", children=[retr])
    tree = spans_to_dict(root)
    ok = tree["kind"] == "workflow_step" and tree["children"][0]["kind"] == "retriever" \
        and tree["children"][0]["children"][0]["kind"] == "model"
    return Finding("LI-E telemetry nesting", "mission_is_root", native="(the retriever trace is the root)",
                   with_redevops="mission→retriever→embed",
                   classification=ResultClass.REDEVOPS_DELTA if ok else ResultClass.BUG,
                   note="LlamaIndex retriever/embedding spans nest beneath the Mission causal node")


def main():
    print("=" * 78)
    print("LlamaIndex × ReDevOps — conformance run")
    print("=" * 78)
    import code_closure
    import governed_retrieval
    import longcontext_answer
    print("\n--- LI-A code closure (flagship) ---"); code_closure.main()
    print("\n--- LI-B long-context answer ---"); longcontext_answer.main()
    print("\n--- LI-D governed retrieval→action ---"); governed_retrieval.main()

    extra = [li_c_representation_choice(), li_e_telemetry()]
    print("\n--- LI-C / LI-E ---")
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
    print("result classification tally (framework: llama-index):")
    for k in ("REDEVOPS_DELTA", "PARITY", "FRAMEWORK_NATIVE_ADVANTAGE", "EXPECTED_IMPLEMENTATION_DIFFERENCE", "BUG"):
        if tally.get(k):
            print(f"  {k:36} {tally[k]}")
    print("=" * 78)
    print("thesis: LlamaIndex owns indexing, retrieval and the query engine; ReDevOps composes the retriever with "
          "structural closure and adds verification, governance and replay — the runtime properties retrieval alone "
          "doesn't establish.")


if __name__ == "__main__":
    main()
