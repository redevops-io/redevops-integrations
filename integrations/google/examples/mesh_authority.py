"""GO-A (flagship) — authority and provenance survive a multi-hop A2A mesh (frameworks plan 7).

Google ADK / A2A's strongest abstraction is agent-to-agent delegation: an agent hands a task to another agent,
which may hand it on again, across service boundaries. So we test the boundary immediately outside a multi-hop
mesh. Two failures are unique to it: (1) TRANSITIVE authority escalation — a 3rd-hop agent doing what an
intermediate hop was never authorized for; (2) loss of PROVENANCE — a coordinator synthesizing a finding it
cannot attribute to an authorized agent. ReDevOps makes authority the intersection of the WHOLE chain (monotone
non-increasing across every hop) and admits a finding into the synthesis only when its provenance chain is
attributable to authorized agents. Real ADK agents form the mesh; the authority/provenance gate is deterministic.

    python examples/mesh_authority.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))

from adapters import ADK_VERSION, AdkAgentCapability, AuthorityChain, Provenance  # noqa: E402
from bundle import AcceptanceBundle  # noqa: E402
from classification import Finding, ResultClass  # noqa: E402

# a research mesh of real ADK agents (constructed; no LLM call in this deterministic authority test)
AGENTS = {n: AdkAgentCapability(name=n) for n in
          ("coordinator", "web_researcher", "scraper", "rogue_relay")}
AUTHORIZED = {"coordinator", "web_researcher", "scraper"}      # rogue_relay is NOT an authorized mesh agent

# root grant the coordinator holds for the research mission
ROOT = AuthorityChain([("coordinator", {"research:read", "web:fetch", "kb:write"})])


def main():
    # --- part 1: transitive authority attenuation across hops ---
    # coordinator → web_researcher (drops kb:write) → scraper (declares web:fetch only)
    chain2 = ROOT.extend("web_researcher", {"research:read", "web:fetch"})       # kb:write dropped here
    chain3 = chain2.extend("scraper", {"research:read", "web:fetch", "kb:write"})  # re-declares kb:write — must NOT regain it
    auth_cases = [
        ("scraper fetches (in scope)", chain3, "web:fetch", True),
        ("scraper writes KB (dropped at hop 2)", chain3, "kb:write", False),   # transitive attenuation
        ("scraper deletes (never granted)", chain3, "kb:delete", False),
        ("web_researcher reads (in scope)", chain2, "research:read", True),
    ]

    # --- part 2: provenance-gated synthesis ---
    findings = [
        ("finding via coordinator→web_researcher", Provenance(["coordinator", "web_researcher"]), True),
        ("finding via coordinator→web_researcher→scraper", Provenance(["coordinator", "web_researcher", "scraper"]), True),
        ("finding relayed by an unknown agent", Provenance(["coordinator", "rogue_relay"]), False),
        ("finding with no provenance", Provenance([]), False),
    ]

    b = AcceptanceBundle(framework="google-adk", framework_version=ADK_VERSION,
                         config={"experiment": "GO-A mesh authority + provenance", "agents": len(AGENTS)})

    print("GO-A multi-hop mesh — real ADK agents, chain-wide attenuation:")
    print(f"  {'case':46} {'chain (hops)':44} {'native':>7} {'redevops':>9}  gt")
    native_bad = wrong = admitted_ok = 0
    for name, chain, perm, should in auth_cases:
        native = True                     # native A2A: the remote agent runs with its own capabilities
        rd = chain.admits(perm)
        native_bad += 1 if (native and not should) else 0
        wrong += 1 if rd != should else 0
        admitted_ok += 1 if (should and rd) else 0
        print(f"  {name:46} {chain.path()[:44]:44} {'runs':>7} {('admit' if rd else 'DENY'):>9}  {should}")

    print("\n  provenance-gated synthesis:")
    prov_bad = prov_wrong = 0
    for name, prov, should in findings:
        native = True                     # native: the coordinator trusts whatever comes back
        rd = prov.attributable(AUTHORIZED)
        prov_bad += 1 if (native and not should) else 0
        prov_wrong += 1 if rd != should else 0
        print(f"  {name:46} {('· '.join(prov.chain) or '—')[:44]:44} {'trusts':>7} {('admit' if rd else 'DENY'):>9}  {should}")

    widen = sum(1 for *_, s in auth_cases if not s)
    inscope = sum(1 for *_, s in auth_cases if s)
    unprov = sum(1 for *_, s in findings if not s)
    b.add(Finding("GO-A transitive authority attenuation", "unauthorized_hops_admitted",
                  native=f"{native_bad}/{widen} run", with_redevops=0 if wrong == 0 else wrong,
                  classification=ResultClass.REDEVOPS_DELTA if wrong == 0 and native_bad > 0 else ResultClass.BUG,
                  note="effective authority is the intersection of the WHOLE delegation chain; a grant dropped at an "
                       "intermediate hop cannot be regained downstream — native A2A would run each remote action"))
    b.add(Finding("GO-A provenance-gated synthesis", "unprovenanced_findings_admitted",
                  native=f"{prov_bad}/{unprov} trusted", with_redevops=0 if prov_wrong == 0 else prov_wrong,
                  classification=ResultClass.REDEVOPS_DELTA if prov_wrong == 0 and prov_bad > 0 else ResultClass.BUG,
                  note="a finding enters the synthesis only when its provenance chain is attributable to authorized "
                       "mesh agents; an unknown relay or a missing chain is excluded"))
    b.add(Finding("GO-A in-scope delegation preserved", "in_scope_admitted",
                  native="(no chain-wide envelope)", with_redevops=f"{admitted_ok}/{inscope}",
                  classification=ResultClass.PARITY if admitted_ok == inscope else ResultClass.BUG,
                  note="legitimately-scoped, authorized-provenance work is admitted across every hop"))

    out = b.write(os.path.join(HERE, "..", "results", "go_a_mesh_authority.json"))
    print(f"\n  transitive attenuation: native runs {native_bad}/{widen} · ReDevOps denies all ({wrong} wrong)")
    print(f"  provenance gate: native trusts {prov_bad}/{unprov} unprovenanced · ReDevOps excludes all ({prov_wrong} wrong)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
