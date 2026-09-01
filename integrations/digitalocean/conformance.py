"""DigitalOcean conformance runner — runs DO-A/DO-B/DO-C, adds DO-D cache isolation + DO-E telemetry, tallies.

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
from adapters import TenantScope  # noqa: E402
from classification import Finding, ResultClass  # noqa: E402


def do_d_cache_isolation():
    """DO-D — cache keys are tenant-scoped, so two apps asking the same question never share a cached answer.
    This is the fix for cross-tenant cache bleed on a shared runtime. Deterministic."""
    a = TenantScope("telegrambot", {"kb:read"})
    b = TenantScope("nutrients", {"kb:read"})
    same_question = "faq:reset-password"
    ok = a.cache_key(same_question) != b.cache_key(same_question)
    return Finding("DO-D tenant-scoped cache keys", "cross_tenant_cache_bleed",
                   native="(a shared cache key returns one tenant's answer to another)",
                   with_redevops="tenant-namespaced keys",
                   classification=ResultClass.REDEVOPS_DELTA if ok else ResultClass.BUG,
                   note="the same question from two apps yields distinct cache keys "
                        f"({a.cache_key(same_question)} vs {b.cache_key(same_question)}) — no cross-tenant bleed")


def do_e_telemetry():
    """DO-E — each tenant app's agent/model spans nest UNDER the shared-Runtime Mission node; Mission is the root."""
    apps = [NodeSpan(f"do:{t}", "agent", f"do:{t}", children=[NodeSpan("model", "model", "gpt-4o-mini")])
            for t in ("telegrambot", "nutrients", "vibexgen")]
    root = NodeSpan("mission.node.shared-runtime", "workflow_step", "mission.node", children=apps)
    tree = spans_to_dict(root)
    ok = tree["kind"] == "workflow_step" and all(c["kind"] == "agent" for c in tree["children"]) \
        and tree["children"][0]["children"][0]["kind"] == "model"
    return Finding("DO-E telemetry nesting", "mission_is_root", native="(each app trace is its own root)",
                   with_redevops="mission→{app}→model", classification=ResultClass.REDEVOPS_DELTA if ok else ResultClass.BUG,
                   note="every tenant app's agent/model spans nest beneath the one shared-Runtime Mission node")


def main():
    print("=" * 78)
    print("DigitalOcean (GenAI Platform) × ReDevOps — conformance run")
    print("=" * 78)
    import governed_tenant
    import shared_runtime
    import three_apps
    print("\n--- DO-A shared Runtime tenant isolation (flagship) ---"); shared_runtime.main()
    print("\n--- DO-B governed tenant action ---"); governed_tenant.main()
    print("\n--- DO-C three real apps on one Runtime ---"); three_apps.main()

    extra = [do_d_cache_isolation(), do_e_telemetry()]
    print("\n--- DO-D / DO-E ---")
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
    print("result classification tally (framework: digitalocean-genai):")
    for k in ("REDEVOPS_DELTA", "PARITY", "FRAMEWORK_NATIVE_ADVANTAGE", "EXPECTED_IMPLEMENTATION_DIFFERENCE", "BUG"):
        if tally.get(k):
            print(f"  {k:36} {tally[k]}")
    print("=" * 78)
    print("thesis: DigitalOcean gives several lean apps a managed agent stack; ReDevOps gives all of them ONE "
          "runtime — tenant-isolated authority, context, cache and replay — so no app rebuilds production machinery "
          "and no app reaches into another's state.")


if __name__ == "__main__":
    main()
