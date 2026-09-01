"""DO-A (flagship) — three DigitalOcean apps share ONE Runtime, tenant-isolated (frameworks plan 7).

DigitalOcean gives several lean apps a managed agent + knowledge-base stack. The differentiator here is not a new
agent framework — it is the SHARED RUNTIME. telegrambot.ai, nutrients.tech and vibexgen.io run on ONE ReDevOps
Runtime that provides context, retry, replay, governance and telemetry once, instead of each app rebuilding them.
The boundary immediately outside a naive shared runtime is TENANT ISOLATION: one app must not read another's
context or exercise another's authority. ReDevOps scopes every access to the acting tenant, deny-by-default across
tenants. Deterministic, so the isolation mechanism is isolated.

    python examples/shared_runtime.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))

from adapters import DO_API, SharedRuntime, TenantScope  # noqa: E402
from bundle import AcceptanceBundle  # noqa: E402
from classification import Finding, ResultClass  # noqa: E402

# the three real DO apps, each a tenant with its own grants + private context namespace
TENANTS = [
    TenantScope("telegrambot", {"chat:reply", "kb:read"}),
    TenantScope("nutrients", {"plan:generate", "kb:read"}),
    TenantScope("vibexgen", {"reel:render", "kb:read"}),
]
RT = SharedRuntime(TENANTS)

# (label, acting tenant, action, resource namespace, ground-truth admit)
CASES = [
    ("telegrambot reads its own KB", "telegrambot", "kb:read", "tenant/telegrambot", True),
    ("nutrients generates its own plan", "nutrients", "plan:generate", "tenant/nutrients", True),
    ("telegrambot reads nutrients' context", "telegrambot", "kb:read", "tenant/nutrients", False),   # cross-tenant read
    ("nutrients renders a vibexgen reel", "nutrients", "reel:render", "tenant/nutrients", False),     # not its grant
    ("vibexgen replies as telegrambot", "vibexgen", "chat:reply", "tenant/vibexgen", False),          # not its grant
    ("vibexgen renders its own reel", "vibexgen", "reel:render", "tenant/vibexgen", True),
]


def main():
    print("DO-A shared Runtime — three DO apps, tenant-isolated:")
    print(f"  {'case':44} {'tenant':13} {'native':>7} {'redevops':>9}  gt")

    b = AcceptanceBundle(framework="digitalocean-genai", framework_version=DO_API,
                         config={"experiment": "DO-A shared runtime tenant isolation", "tenants": len(TENANTS)})
    native_bad = wrong = admitted_ok = 0
    for label, tenant, action, ns, should in CASES:
        native = True                     # a naive shared runtime does not scope the access
        rd, reason = RT.access(tenant, action, ns)
        native_bad += 1 if (native and not should) else 0
        wrong += 1 if rd != should else 0
        admitted_ok += 1 if (should and rd) else 0
        print(f"  {label:44} {tenant:13} {'runs':>7} {('admit' if rd else 'DENY'):>9}  {should}"
              f"   {'' if rd == should else reason}")

    bad = sum(1 for *_, s in CASES if not s)
    good = sum(1 for *_, s in CASES if s)
    # shared machinery: what N apps would each rebuild vs what the one runtime provides
    rebuilt = RT.machinery_rebuilt_per_app()
    once = RT.machinery_provided_once()

    b.add(Finding("DO-A cross-tenant isolation", "cross_tenant_accesses_admitted",
                  native=f"{native_bad}/{bad} run", with_redevops=0 if wrong == 0 else wrong,
                  classification=ResultClass.REDEVOPS_DELTA if wrong == 0 and native_bad > 0 else ResultClass.BUG,
                  note="every access is scoped to the acting tenant; a cross-tenant read or an out-of-grant action is "
                       "denied by default, while a naive shared runtime would run each"))
    b.add(Finding("DO-A in-tenant work preserved", "in_scope_admitted", native="(no tenant scoping)",
                  with_redevops=f"{admitted_ok}/{good}",
                  classification=ResultClass.PARITY if admitted_ok == good else ResultClass.BUG,
                  note="each app's own in-grant, own-namespace work is admitted — isolation blocks the cross-tenant "
                       "without blocking the app"))
    b.add(Finding("DO-A shared machinery, once", "capabilities_provided",
                  native=f"{rebuilt} (rebuilt per app: {len(TENANTS)}×{once})", with_redevops=once,
                  classification=ResultClass.REDEVOPS_DELTA,
                  note=f"context/retry/replay/governance/telemetry are provided once by the shared Runtime instead "
                       f"of rebuilt in each of the {len(TENANTS)} apps"))

    out = b.write(os.path.join(HERE, "..", "results", "do_a_shared_runtime.json"))
    print(f"\n  cross-tenant: native runs {native_bad}/{bad} · ReDevOps denies all ({wrong} wrong)")
    print(f"  in-tenant admitted: {admitted_ok}/{good} · shared machinery {once} once vs {rebuilt} rebuilt per app")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
