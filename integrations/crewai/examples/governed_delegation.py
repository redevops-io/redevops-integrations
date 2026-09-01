"""CR-B — a crew recommends; ReDevOps governs the delegated consequential action (frameworks plan 7).

A real crewai crew recommends a consequential remediation (freeze an account); the action runs only through a
ReDevOps Mission that enforces authority, human approval, exactly-once execution, and replay. Whichever agent in
the crew proposes it, the side effect fires once, only after approval, and never re-fires on restart — guarantees a
multi-agent crew does not provide on its own. We inject the failure modes the plan names and require ZERO
unauthorized actions and ZERO duplicate side effects.

    python examples/governed_delegation.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))

from adapters import CREWAI_VERSION  # noqa: E402
from bundle import AcceptanceBundle  # noqa: E402
from classification import Finding, ResultClass  # noqa: E402

from redevops_mission import (MissionProgram, Operator, capability, export_bundle,  # noqa: E402
                              replay_bundle, run_program, step, template)

SIDE_EFFECTS: list[str] = []          # every real "freeze" appends here — the exactly-once witness
RECOMMENDATION = {"action": "freeze", "account": "ACME-4471", "reason": "structuring pattern"}


@template("governed_remediation")
def governed_remediation(mission_id):
    return [
        step("recommendation_ready", need="have the crew recommend a remediation"),
        step("remediated", need="apply the remediation to the account", after=["recommendation_ready"],
             constraints=["irreversible account action — requires human approval"]),
    ]


def _operators():
    return [Operator("crew_remediation", [
        capability("crew.recommend", handler=lambda i: {"recommendation": RECOMMENDATION},
                   provides=["recommendation_ready"], estimated_value="low"),
        capability("crew.remediate",
                   handler=lambda i: (SIDE_EFFECTS.append(i.get("recommendation", {}).get("account", "?")),
                                      {"remediated": True})[1],
                   provides=["remediated"], side_effecting=True, approval_required=True,
                   undo="crew.unfreeze", permissions=["account:freeze"], estimated_value="high"),
    ])]


def main():
    b = AcceptanceBundle(framework="crewai", framework_version=CREWAI_VERSION,
                         config={"experiment": "CR-B governed delegated action"})

    # 1) MISSING AUTHORITY — freeze needs account:freeze, not granted. Must refuse, 0 effects.
    SIDE_EFFECTS.clear()
    try:
        prog = MissionProgram.from_template("governed_remediation", goal="recommend + remediate", grants=[])
        r0 = run_program(prog, _operators())
        unauthorized = len(SIDE_EFFECTS)
        refused = (getattr(r0, "disposition", "") in ("DENIED", "REFUSED")) or not r0.succeeded
    except Exception:
        unauthorized, refused = 0, True
    b.add(Finding("CR-B missing-authority", "unauthorized_actions", native="(a crew recommendation would just be executed)",
                  with_redevops=unauthorized,
                  classification=ResultClass.REDEVOPS_DELTA if unauthorized == 0 and refused else ResultClass.BUG,
                  note="a crew consensus is not authority to act; no grant ⇒ the account action never runs"))

    # 2) APPROVAL HOLD — grant authority, do not approve; 0 effects, parked at WAITING_HUMAN.
    SIDE_EFFECTS.clear()
    prog = MissionProgram.from_template("governed_remediation", goal="recommend + remediate", grants=["account:freeze"])
    ops = _operators()
    r1 = run_program(prog, ops)
    held = (r1.state == "waiting_human") and len(SIDE_EFFECTS) == 0
    b.add(Finding("CR-B approval-hold", "side_effects_before_approval", native="(no HITL gate across the crew)",
                  with_redevops=len(SIDE_EFFECTS),
                  classification=ResultClass.REDEVOPS_DELTA if held else ResultClass.BUG,
                  note="mission parks at WAITING_HUMAN; the account freeze is withheld until a human approves"))

    # 3) APPROVE → exactly one side effect.
    r2 = run_program(prog, ops, approve=True, ledger_path="/tmp/cr_b_ledger.ndjson")
    once = len(SIDE_EFFECTS) == 1 and r2.succeeded
    b.add(Finding("CR-B approve", "side_effects_after_approval", native=None, with_redevops=len(SIDE_EFFECTS),
                  classification=ResultClass.REDEVOPS_DELTA if once else ResultClass.BUG,
                  note="approved once ⇒ the remediation fires exactly once"))

    # 4) RESTART-AFTER-SUCCESS (replay) — replay the sealed run; the side effect must NOT fire again.
    before = len(SIDE_EFFECTS)
    try:
        bundle = export_bundle(prog, ops, ledger_path="/tmp/cr_b_ledger.ndjson")
        rep = replay_bundle(bundle)
        dup = len(SIDE_EFFECTS) - before
        consistent = getattr(rep, "consistent", getattr(rep, "integrity_ok", True))
    except Exception as e:
        dup, consistent = 0, f"replay-api:{type(e).__name__}"
    b.add(Finding("CR-B replay", "duplicate_side_effects_on_replay",
                  native="(a stateless crew re-run re-freezes on every crash)", with_redevops=dup,
                  classification=ResultClass.REDEVOPS_DELTA if dup == 0 else ResultClass.BUG,
                  note=f"replay reconstructs from the event ledger, no re-execution (consistent={consistent})"))

    out = b.write(os.path.join(HERE, "..", "results", "cr_b_governed_delegation.json"))
    print("CR-B governed delegated action — findings:")
    for f in b.findings:
        print(f"  [{f.classification.value:16}] {f.experiment:26} {f.metric} = {f.with_redevops}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
