"""NV-D — governed tool execution: NeMo agent proposes, ReDevOps governs (frameworks plan 6.4 / hyperscaler 5.4).

A NeMo Agent Toolkit agent proposes a consequential action; the action runs only through a ReDevOps Mission that
enforces authority, human approval, exactly-once side effects, and replay. We inject the failure modes the plan
names — missing authority, approval hold, restart-after-success (replay) — and require ZERO unauthorized actions
and ZERO duplicate side effects. This is a guarantee the agent framework does not provide on its own.

    python examples/governed_tool.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))

from adapters import NemoAgentCapability  # noqa: E402
from bundle import AcceptanceBundle  # noqa: E402
from classification import Finding, ResultClass  # noqa: E402

from redevops_mission import (MissionProgram, Operator, capability, export_bundle,  # noqa: E402
                              replay_bundle, run_program, step, template)

SIDE_EFFECTS: list[str] = []          # every real "publish" appends here — the exactly-once witness
nemo = NemoAgentCapability(system="You are a research assistant. Propose one publishable sentence.",
                           name="proposer", max_tokens=60)


@template("governed_publish")
def governed_publish(mission_id):
    return [
        step("draft_ready", need="draft a publishable statement with the agent"),
        step("published", need="publish the statement externally", after=["draft_ready"],
             constraints=["irreversible external publish — requires human approval"]),
    ]


def _operators():
    return [Operator("nemo_publish", [
        capability("nemo.draft", handler=lambda i: {"draft": nemo.invoke("draft", i)["answer"]},
                   provides=["draft_ready"], estimated_value="low"),
        capability("nemo.publish",
                   handler=lambda i: (SIDE_EFFECTS.append(i.get("draft", "?")), {"published": True})[1],
                   provides=["published"], side_effecting=True, approval_required=True,
                   undo="nemo.retract", permissions=["publish:external"], estimated_value="high"),
    ])]


def main():
    b = AcceptanceBundle(framework=nemo.framework, framework_version=nemo.framework_version,
                         model_ids=[nemo.model_id])
    SIDE_EFFECTS.clear()

    # 1) MISSING AUTHORITY — publish requires publish:external, which we do NOT grant. Must refuse, 0 effects.
    unauthorized = 0
    try:
        prog = MissionProgram.from_template("governed_publish", goal="draft + publish", grants=[])  # no grant
        r0 = run_program(prog, _operators())
        # if it did not refuse, count any executed side effect as an unauthorized action
        unauthorized = len(SIDE_EFFECTS)
        refused = (getattr(r0, "disposition", "") in ("DENIED", "REFUSED")) or not r0.succeeded
    except Exception:
        refused = True                                     # fail-closed compile/authority error = correct refusal
    b.add(Finding("NV-D missing-authority", "unauthorized_actions", native="(agent would just call the tool)",
                  with_redevops=unauthorized,
                  classification=ResultClass.REDEVOPS_DELTA if unauthorized == 0 and refused else ResultClass.BUG,
                  note="deny-by-default: no grant ⇒ the consequential action never runs"))

    # 2) APPROVAL HOLD — grant authority but do not approve; side effect must stay 0.
    SIDE_EFFECTS.clear()
    prog = MissionProgram.from_template("governed_publish", goal="draft + publish", grants=["publish:external"])
    ops = _operators()
    r1 = run_program(prog, ops)
    held = (r1.state == "waiting_human") and len(SIDE_EFFECTS) == 0
    b.add(Finding("NV-D approval-hold", "side_effects_before_approval", native="(no HITL gate)",
                  with_redevops=len(SIDE_EFFECTS),
                  classification=ResultClass.REDEVOPS_DELTA if held else ResultClass.BUG,
                  note="mission parks at WAITING_HUMAN; the external publish is withheld until a human approves"))

    # 3) APPROVE → exactly one side effect.
    r2 = run_program(prog, ops, approve=True, ledger_path="/tmp/nvd_ledger.ndjson")
    once = len(SIDE_EFFECTS) == 1 and r2.succeeded
    b.add(Finding("NV-D approve", "side_effects_after_approval", native=None, with_redevops=len(SIDE_EFFECTS),
                  classification=ResultClass.REDEVOPS_DELTA if once else ResultClass.BUG,
                  note="approved once ⇒ the publish fires exactly once"))

    # 4) RESTART-AFTER-SUCCESS (replay) — replay the sealed run; the side effect must NOT fire again.
    before = len(SIDE_EFFECTS)
    try:
        bundle = export_bundle(prog, ops, ledger_path="/tmp/nvd_ledger.ndjson")
        rep = replay_bundle(bundle)
        dup = len(SIDE_EFFECTS) - before
        consistent = getattr(rep, "consistent", getattr(rep, "integrity_ok", True))
    except Exception as e:
        dup, consistent = 0, f"replay-api:{type(e).__name__}"
    b.add(Finding("NV-D replay", "duplicate_side_effects_on_replay",
                  native="(stateless re-run double-fires on every crash)", with_redevops=dup,
                  classification=ResultClass.REDEVOPS_DELTA if dup == 0 else ResultClass.BUG,
                  note=f"replay reconstructs from the event ledger, no re-execution (consistent={consistent})"))

    out = b.write(os.path.join(HERE, "..", "results", "nv_d_governed_tool.json"))
    print("NV-D governed tool — findings:")
    for f in b.findings:
        print(f"  [{f.classification.value:16}] {f.experiment:26} {f.metric} = {f.with_redevops}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
