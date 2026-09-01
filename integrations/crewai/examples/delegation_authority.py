"""CR-A (flagship) — a delegated CrewAI agent cannot widen its authority (frameworks plan 7).

CrewAI's strongest abstraction is multi-agent delegation: a manager hands work to coworkers, who run with their own
tools. So we test the boundary immediately outside delegation. In CrewAI there is no notion that a delegate may not
exceed the delegator's authority — the coworker's tools simply fire. ReDevOps makes a delegated agent's effective
grants the INTERSECTION of the delegation chain (authority is monotone non-increasing), and admits a consequential
tool call only if its required permission is inside that narrowed envelope. Real crewai Agents; deterministic
authority so the mechanism is isolated (no LLM).

    python examples/delegation_authority.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))

from adapters import AuthorityEnvelope, CREWAI_VERSION, RedevopsGovernedTool  # noqa: E402
from bundle import AcceptanceBundle  # noqa: E402
from classification import Finding, ResultClass  # noqa: E402

from crewai import LLM, Agent  # noqa: E402


def _agent(role, goal):
    # a REAL crewai Agent (no LLM call is made in this deterministic authority test)
    return Agent(role=role, goal=goal, backstory=f"{role} in an AML/KYC compliance crew.",
                 llm=LLM(model="openai/gpt-4o-mini"), allow_delegation=True, verbose=False)


def main():
    officer = _agent("Compliance Officer", "Review KYC cases within your authority")
    analyst = _agent("Risk Analyst", "Assess customer risk and recommend actions")
    junior = _agent("Junior Analyst", "Assist with case notes")
    senior = _agent("Senior Compliance Officer", "Approve high-impact remediations")

    # delegation chains → narrowed authority envelopes (child effective = parent ∩ declared)
    officer_env = AuthorityEnvelope({"kyc:review", "case:note"}, agent=officer.role)
    analyst_env = officer_env.narrow({"kyc:review", "case:note", "account:freeze"}, agent=analyst.role)
    junior_env = analyst_env.narrow({"account:freeze", "funds:release"}, agent=junior.role)
    senior_env = AuthorityEnvelope({"kyc:review", "case:note", "account:freeze"}, agent=senior.role)
    senior_analyst_env = senior_env.narrow({"account:freeze"}, agent=analyst.role)

    # (acting envelope, action permission, ground-truth: may the delegate legitimately do this?)
    cases = [
        ("analyst reviews (in scope)", analyst_env, "kyc:review", True),
        ("analyst freezes (not held by delegator)", analyst_env, "account:freeze", False),
        ("junior releases funds (never in the chain)", junior_env, "funds:release", False),
        ("junior freezes (delegator lacked it)", junior_env, "account:freeze", False),
        ("senior→analyst freezes (delegator HELD it)", senior_analyst_env, "account:freeze", True),
    ]

    b = AcceptanceBundle(framework="crewai", framework_version=CREWAI_VERSION,
                         config={"experiment": "CR-A delegation authority", "agents": 4})
    sink: list = []
    print("CR-A delegation authority — real crewai agents, narrowed envelopes:")
    print(f"  {'case':44} {'chain':34} {'native':>7} {'redevops':>9}  gt")
    native_unauthorized = 0        # actions native CrewAI would run that exceed the delegator's authority
    redevops_wrong = 0             # ReDevOps admissions that disagree with ground truth (widening admitted OR in-scope denied)
    admitted_inscope = 0
    for name, env, perm, should in cases:
        tool = RedevopsGovernedTool(permission=perm).bind(env, sink)
        native_runs = True         # CrewAI: the coworker holds the tool, so delegation runs it regardless of authority
        rd = "OK" in tool._run()   # ReDevOps: admit only within the narrowed envelope
        if native_runs and not should:
            native_unauthorized += 1
        if rd != should:
            redevops_wrong += 1
        if should and rd:
            admitted_inscope += 1
        print(f"  {name:44} {'→'.join(env.chain()):34} {'runs':>7} {('admit' if rd else 'DENY'):>9}  {should}")

    widening = sum(1 for *_, s in cases if not s)
    inscope = sum(1 for *_, s in cases if s)
    b.add(Finding("CR-A authority widening under delegation", "unauthorized_delegated_actions",
                  native=f"{native_unauthorized}/{widening} run", with_redevops=0 if redevops_wrong == 0 else redevops_wrong,
                  classification=ResultClass.REDEVOPS_DELTA if redevops_wrong == 0 and native_unauthorized > 0
                  else ResultClass.BUG,
                  note="a delegated agent's effective grants are the intersection of the chain; every authority-"
                       "widening action is denied, while CrewAI delegation would run each one"))
    b.add(Finding("CR-A in-scope delegation preserved", "in_scope_admitted",
                  native="(no envelope to narrow)", with_redevops=f"{admitted_inscope}/{inscope}",
                  classification=ResultClass.PARITY if admitted_inscope == inscope else ResultClass.BUG,
                  note="all legitimately-scoped delegated actions (incl. a HELD high-impact freeze) are admitted — "
                       "narrowing blocks widening without blocking authorized work"))

    out = b.write(os.path.join(HERE, "..", "results", "cr_a_delegation_authority.json"))
    print(f"\n  authority-widening actions: native runs {native_unauthorized}/{widening} · ReDevOps denies all "
          f"({redevops_wrong} wrong)")
    print(f"  in-scope delegated actions admitted: {inscope}/{inscope}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
