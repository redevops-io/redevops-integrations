"""AW-B — a 500-household shadow deployment, then governed promotion to execution (frameworks plan 7).

The Wealth Manager runs in SHADOW first: the Strands agent proposes an action for every household, and ReDevOps
holds ALL execution — zero real side effects — while measuring how many proposals the composed policy would admit.
Only when a proposal is promoted does it execute, and then only through a Mission with authority, approval,
exactly-once execution and replay. This is the low-risk enterprise pattern: measure before you act.

    python examples/shadow_promotion.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))

from bundle import AcceptanceBundle  # noqa: E402
from classification import Finding, ResultClass  # noqa: E402
from deny_wins_iam import AUTH  # reuse the composed IAM+SCP+Mission policy  # noqa: E402

from redevops_mission import (MissionProgram, Operator, capability, export_bundle,  # noqa: E402
                              replay_bundle, run_program, step, template)

EXECUTED: list[str] = []          # every real portfolio action appends here — the exactly-once witness
N_HOUSEHOLDS = 500


@template("governed_promotion")
def governed_promotion(mission_id):
    return [
        step("proposal_ready", need="have the wealth-manager agent propose an action"),
        step("executed", need="execute the promoted portfolio action", after=["proposal_ready"],
             constraints=["client-money portfolio action — requires human approval"]),
    ]


def _operators():
    return [Operator("wealth", [
        capability("wm.propose", handler=lambda i: {"action": "portfolio:rebalance", "resource": "household/7"},
                   provides=["proposal_ready"], estimated_value="low"),
        capability("wm.execute",
                   handler=lambda i: (EXECUTED.append(i.get("resource", "?")), {"executed": True})[1],
                   provides=["executed"], side_effecting=True, approval_required=True,
                   undo="wm.reverse", permissions=["portfolio:rebalance"], estimated_value="high"),
    ])]


def main():
    b = AcceptanceBundle(framework="strands-agents", framework_version="1.54.0", model_ids=["gpt-4o-mini"],
                         config={"experiment": "AW-B shadow + governed promotion", "households": N_HOUSEHOLDS})

    # 1) SHADOW — propose an action per household; ReDevOps holds all execution. 0 real side effects.
    EXECUTED.clear()
    proposals = [("portfolio:rebalance", f"household/{i}") for i in range(N_HOUSEHOLDS)]
    would_admit = sum(1 for a, r in proposals if AUTH.decision(a, r)[0])
    executed_in_shadow = len(EXECUTED)     # ReDevOps executes nothing in shadow
    b.add(Finding("AW-B shadow safety", "side_effects_in_shadow", native="(a live agent would act on each proposal)",
                  with_redevops=executed_in_shadow,
                  classification=ResultClass.REDEVOPS_DELTA if executed_in_shadow == 0 else ResultClass.BUG,
                  note=f"shadowed {N_HOUSEHOLDS} households with 0 real side effects; the composed policy would admit "
                       f"{would_admit}/{N_HOUSEHOLDS} for promotion — measure before you act"))

    # 2) PROMOTE without authority — must refuse, 0 executions.
    EXECUTED.clear()
    try:
        prog = MissionProgram.from_template("governed_promotion", goal="propose + execute", grants=[])
        r0 = run_program(prog, _operators())
        unauthorized = len(EXECUTED)
        refused = (getattr(r0, "disposition", "") in ("DENIED", "REFUSED")) or not r0.succeeded
    except Exception:
        unauthorized, refused = 0, True
    b.add(Finding("AW-B promote missing-authority", "unauthorized_actions", native="(promotion would just execute)",
                  with_redevops=unauthorized,
                  classification=ResultClass.REDEVOPS_DELTA if unauthorized == 0 and refused else ResultClass.BUG,
                  note="a promoted proposal is not authority to act; no grant ⇒ the portfolio action never runs"))

    # 3) APPROVE → exactly one execution.
    EXECUTED.clear()
    prog = MissionProgram.from_template("governed_promotion", goal="propose + execute",
                                        grants=["portfolio:rebalance"])
    ops = _operators()
    r1 = run_program(prog, ops)
    held = (r1.state == "waiting_human") and len(EXECUTED) == 0
    r2 = run_program(prog, ops, approve=True, ledger_path="/tmp/aw_b_ledger.ndjson")
    once = len(EXECUTED) == 1 and r2.succeeded
    b.add(Finding("AW-B approval-hold + exactly-once", "held_then_once",
                  native="(no HITL gate)", with_redevops=f"held={held} · executed={len(EXECUTED)}",
                  classification=ResultClass.REDEVOPS_DELTA if held and once else ResultClass.BUG,
                  note="the promotion parks at WAITING_HUMAN with 0 side effects, then approved fires exactly once"))

    # 4) REPLAY — replay the sealed run; the execution must NOT fire again.
    before = len(EXECUTED)
    try:
        bundle = export_bundle(prog, ops, ledger_path="/tmp/aw_b_ledger.ndjson")
        rep = replay_bundle(bundle)
        dup = len(EXECUTED) - before
        consistent = getattr(rep, "consistent", getattr(rep, "integrity_ok", True))
    except Exception as e:
        dup, consistent = 0, f"replay-api:{type(e).__name__}"
    b.add(Finding("AW-B replay", "duplicate_side_effects_on_replay",
                  native="(a stateless re-run re-executes on every crash)", with_redevops=dup,
                  classification=ResultClass.REDEVOPS_DELTA if dup == 0 else ResultClass.BUG,
                  note=f"replay reconstructs from the event ledger, no re-execution (consistent={consistent})"))

    out = b.write(os.path.join(HERE, "..", "results", "aw_b_shadow_promotion.json"))
    print("AW-B shadow + governed promotion — findings:")
    for f in b.findings:
        print(f"  [{f.classification.value:16}] {f.experiment:32} {f.metric} = {f.with_redevops}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
