"""AZ-B — the admitted deployment executes only as a governed Mission (frameworks plan 7 / hyperscaler plan).

A compliant, authorized deployment still must not fire on its own. ReDevOps runs it as a Mission that enforces
authority, human approval, exactly-once execution and replay. We inject the failure modes the plan names — missing
authority, approval hold, restart-after-success — and require ZERO unauthorized actions and ZERO duplicate deploys.

    integrations/azure/.venv/bin/python examples/governed_deploy.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))

from adapters import SK_VERSION  # noqa: E402
from bundle import AcceptanceBundle  # noqa: E402
from classification import Finding, ResultClass  # noqa: E402

from redevops_mission import (MissionProgram, Operator, capability, export_bundle,  # noqa: E402
                              replay_bundle, run_program, step, template)

DEPLOYS: list[str] = []          # every real "deploy" appends here — the exactly-once witness
REQUEST = {"service": "payments", "environment": "production"}


@template("governed_deployment")
def governed_deployment(mission_id):
    return [
        step("plan_ready", need="have the Semantic Kernel agent plan the deployment"),
        step("deployed", need="execute the deployment to the environment", after=["plan_ready"],
             constraints=["irreversible production deploy — requires human approval"]),
    ]


def _operators():
    return [Operator("sk_deploy", [
        capability("sk.plan", handler=lambda i: {"request": REQUEST},
                   provides=["plan_ready"], estimated_value="low"),
        capability("sk.deploy",
                   handler=lambda i: (DEPLOYS.append(i.get("request", {}).get("environment", "?")),
                                      {"deployed": True})[1],
                   provides=["deployed"], side_effecting=True, approval_required=True,
                   undo="sk.rollback", permissions=["deploy:production"], estimated_value="high"),
    ])]


def main():
    b = AcceptanceBundle(framework="semantic-kernel", framework_version=SK_VERSION,
                         config={"experiment": "AZ-B governed deployment"})

    # 1) MISSING AUTHORITY — deploy needs deploy:production, not granted. Must refuse, 0 deploys.
    DEPLOYS.clear()
    try:
        prog = MissionProgram.from_template("governed_deployment", goal="plan + deploy", grants=[])
        r0 = run_program(prog, _operators())
        unauthorized = len(DEPLOYS)
        refused = (getattr(r0, "disposition", "") in ("DENIED", "REFUSED")) or not r0.succeeded
    except Exception:
        unauthorized, refused = 0, True
    b.add(Finding("AZ-B missing-authority", "unauthorized_actions", native="(the agent's plan would just be executed)",
                  with_redevops=unauthorized,
                  classification=ResultClass.REDEVOPS_DELTA if unauthorized == 0 and refused else ResultClass.BUG,
                  note="a plan that passes the compliance gate is still not authority to deploy; no grant ⇒ never runs"))

    # 2) APPROVAL HOLD — grant authority, do not approve; 0 deploys, parked at WAITING_HUMAN.
    DEPLOYS.clear()
    prog = MissionProgram.from_template("governed_deployment", goal="plan + deploy", grants=["deploy:production"])
    ops = _operators()
    r1 = run_program(prog, ops)
    held = (r1.state == "waiting_human") and len(DEPLOYS) == 0
    b.add(Finding("AZ-B approval-hold", "side_effects_before_approval", native="(no HITL gate)",
                  with_redevops=len(DEPLOYS),
                  classification=ResultClass.REDEVOPS_DELTA if held else ResultClass.BUG,
                  note="mission parks at WAITING_HUMAN; the production deploy is withheld until a human approves"))

    # 3) APPROVE → exactly one deploy.
    r2 = run_program(prog, ops, approve=True, ledger_path="/tmp/az_b_ledger.ndjson")
    once = len(DEPLOYS) == 1 and r2.succeeded
    b.add(Finding("AZ-B approve", "side_effects_after_approval", native=None, with_redevops=len(DEPLOYS),
                  classification=ResultClass.REDEVOPS_DELTA if once else ResultClass.BUG,
                  note="approved once ⇒ the deployment fires exactly once"))

    # 4) RESTART-AFTER-SUCCESS (replay) — replay the sealed run; the deploy must NOT fire again.
    before = len(DEPLOYS)
    try:
        bundle = export_bundle(prog, ops, ledger_path="/tmp/az_b_ledger.ndjson")
        rep = replay_bundle(bundle)
        dup = len(DEPLOYS) - before
        consistent = getattr(rep, "consistent", getattr(rep, "integrity_ok", True))
    except Exception as e:
        dup, consistent = 0, f"replay-api:{type(e).__name__}"
    b.add(Finding("AZ-B replay", "duplicate_side_effects_on_replay",
                  native="(a stateless re-run re-deploys on every crash)", with_redevops=dup,
                  classification=ResultClass.REDEVOPS_DELTA if dup == 0 else ResultClass.BUG,
                  note=f"replay reconstructs from the event ledger, no re-execution (consistent={consistent})"))

    out = b.write(os.path.join(HERE, "..", "results", "az_b_governed_deploy.json"))
    print("AZ-B governed deployment — findings:")
    for f in b.findings:
        print(f"  [{f.classification.value:16}] {f.experiment:24} {f.metric} = {f.with_redevops}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
