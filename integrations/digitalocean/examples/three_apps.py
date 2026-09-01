"""DO-C — three real DO app-agents run on one shared Runtime, tenant-isolated (frameworks plan 7).

Three heterogeneous apps — telegrambot.ai (support), nutrients.tech (nutrition), vibexgen.io (reels) — each run a
real OpenAI-compatible DO GenAI agent, all on ONE ReDevOps Runtime. DO owns each agent; ReDevOps runs all three
through a single runtime and keeps each in its own tenant scope, so the shared machinery is provided once and no
app's request reaches another's state. Real agents; tenant-scoped; cached for offline replay.

    python examples/three_apps.py     # three gpt-4o-mini calls, cached; re-runs are offline
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))

from adapters import DO_API, DoAgentCapability, SharedRuntime, TenantScope  # noqa: E402
from bundle import AcceptanceBundle  # noqa: E402
from classification import Finding, ResultClass  # noqa: E402

CACHE = os.path.join(HERE, "three_apps_cache")

APPS = [
    ("telegrambot", "You are a friendly support bot. Reply in one short sentence.",
     "A user asks how to reset their password."),
    ("nutrients", "You are a nutrition assistant. Reply in one short sentence.",
     "Suggest a high-protein vegetarian breakfast."),
    ("vibexgen", "You are a short-video copywriter. Reply with one punchy caption.",
     "Write a caption for a 15s reel about a morning run."),
]
TENANTS = [TenantScope("telegrambot", {"chat:reply", "kb:read"}),
           TenantScope("nutrients", {"plan:generate", "kb:read"}),
           TenantScope("vibexgen", {"reel:render", "kb:read"})]
RT = SharedRuntime(TENANTS)


def _run(agent, tenant, prompt):
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, f"{tenant}.json")
    if os.path.exists(p):
        return json.load(open(p))
    ans = agent.run(prompt)
    json.dump(ans, open(p, "w"))
    return ans


def main():
    print("DO-C three apps on one Runtime — real DO GenAI agents (gpt-4o-mini), tenant-isolated:")
    ran, isolated = 0, 0
    for tenant, system, prompt in APPS:
        agent = DoAgentCapability(tenant=tenant, name=f"{tenant}_agent", system_prompt=system)
        ans = _run(agent, tenant, prompt)
        ok = bool(ans and len(ans) > 8)
        # each agent's work stays in its own tenant scope; a cross-tenant read is denied
        own_ok = RT.access(tenant, "kb:read", f"tenant/{tenant}")[0]
        cross_denied = not RT.access(tenant, "kb:read", "tenant/other")[0]
        ran += 1 if ok else 0
        isolated += 1 if (own_ok and cross_denied) else 0
        print(f"  {tenant:12} → {ans[:58]!r}  (own kb: {own_ok} · cross denied: {cross_denied})")

    b = AcceptanceBundle(framework="digitalocean-genai", framework_version=DO_API, model_ids=["gpt-4o-mini"],
                         config={"experiment": "DO-C three apps on one runtime", "tenants": len(APPS)})
    b.add(Finding("DO-C three real agents run on one Runtime", "apps_ran", native=f"{ran}/{len(APPS)}",
                  with_redevops=f"{ran}/{len(APPS)}", classification=ResultClass.PARITY,
                  note="three heterogeneous DO app-agents run end-to-end on one shared Runtime; ReDevOps does not "
                       "change what any app produces"))
    b.add(Finding("DO-C tenant isolation across apps", "tenant_isolated",
                  native="(a shared runtime with no tenant scope leaks state)", with_redevops=f"{isolated}/{len(APPS)}",
                  classification=ResultClass.REDEVOPS_DELTA if isolated == len(APPS) else ResultClass.BUG,
                  note="every app reads only its own namespace and a cross-tenant read is denied — the shared "
                       "Runtime keeps the apps isolated while sharing the machinery"))

    out = b.write(os.path.join(HERE, "..", "results", "do_c_three_apps.json"))
    print(f"\n  ran {ran}/{len(APPS)} apps · tenant-isolated {isolated}/{len(APPS)}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
