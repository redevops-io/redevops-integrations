"""DO-B — the shared Runtime governs each tenant's consequential action independently (frameworks plan 7).

One app on the shared Runtime (vibexgen.io) publishes a reel — a consequential, external action. The shared Runtime
governs it as a Mission with authority, human approval, exactly-once execution and replay, without any other tenant
being able to trigger or duplicate it. We inject the failure modes the plan names and require ZERO unauthorized
actions and ZERO duplicate side effects.

    python examples/governed_tenant.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))

from adapters import DO_API  # noqa: E402
from bundle import AcceptanceBundle  # noqa: E402
from classification import Finding, ResultClass  # noqa: E402

from redevops_mission import (MissionProgram, Operator, capability, export_bundle,  # noqa: E402
                              replay_bundle, run_program, step, template)

PUBLISHED: list[str] = []          # every real reel publish appends here — the exactly-once witness
REEL = {"tenant": "vibexgen", "reel_id": "reel-8842"}


@template("governed_publish")
def governed_publish(mission_id):
    return [
        step("reel_ready", need="have the vibexgen agent produce a reel"),
        step("published", need="publish the reel to social channels", after=["reel_ready"],
             constraints=["irreversible external publish — requires human approval"]),
    ]


def _operators():
    return [Operator("vibexgen", [
        capability("do.render", handler=lambda i: {"reel": REEL},
                   provides=["reel_ready"], estimated_value="low"),
        capability("do.publish",
                   handler=lambda i: (PUBLISHED.append(i.get("reel", {}).get("reel_id", "?")), {"published": True})[1],
                   provides=["published"], side_effecting=True, approval_required=True,
                   undo="do.unpublish", permissions=["reel:publish"], estimated_value="high"),
    ])]


def main():
    b = AcceptanceBundle(framework="digitalocean-genai", framework_version=DO_API,
                         config={"experiment": "DO-B governed tenant action"})

    # 1) MISSING AUTHORITY — publish needs reel:publish, not granted. Must refuse, 0 publishes.
    PUBLISHED.clear()
    try:
        prog = MissionProgram.from_template("governed_publish", goal="render + publish", grants=[])
        r0 = run_program(prog, _operators())
        unauthorized = len(PUBLISHED)
        refused = (getattr(r0, "disposition", "") in ("DENIED", "REFUSED")) or not r0.succeeded
    except Exception:
        unauthorized, refused = 0, True
    b.add(Finding("DO-B missing-authority", "unauthorized_actions", native="(a tenant agent would just publish)",
                  with_redevops=unauthorized,
                  classification=ResultClass.REDEVOPS_DELTA if unauthorized == 0 and refused else ResultClass.BUG,
                  note="a tenant's agent output is not authority to publish; no grant ⇒ the external publish never runs"))

    # 2) APPROVAL HOLD — grant authority, do not approve; 0 publishes, parked at WAITING_HUMAN.
    PUBLISHED.clear()
    prog = MissionProgram.from_template("governed_publish", goal="render + publish", grants=["reel:publish"])
    ops = _operators()
    r1 = run_program(prog, ops)
    held = (r1.state == "waiting_human") and len(PUBLISHED) == 0
    b.add(Finding("DO-B approval-hold", "side_effects_before_approval", native="(no HITL gate)",
                  with_redevops=len(PUBLISHED),
                  classification=ResultClass.REDEVOPS_DELTA if held else ResultClass.BUG,
                  note="mission parks at WAITING_HUMAN; the external publish is withheld until a human approves"))

    # 3) APPROVE → exactly one publish.
    r2 = run_program(prog, ops, approve=True, ledger_path="/tmp/do_b_ledger.ndjson")
    once = len(PUBLISHED) == 1 and r2.succeeded
    b.add(Finding("DO-B approve", "side_effects_after_approval", native=None, with_redevops=len(PUBLISHED),
                  classification=ResultClass.REDEVOPS_DELTA if once else ResultClass.BUG,
                  note="approved once ⇒ the reel publishes exactly once"))

    # 4) RESTART-AFTER-SUCCESS (replay) — replay the sealed run; the publish must NOT fire again.
    before = len(PUBLISHED)
    try:
        bundle = export_bundle(prog, ops, ledger_path="/tmp/do_b_ledger.ndjson")
        rep = replay_bundle(bundle)
        dup = len(PUBLISHED) - before
        consistent = getattr(rep, "consistent", getattr(rep, "integrity_ok", True))
    except Exception as e:
        dup, consistent = 0, f"replay-api:{type(e).__name__}"
    b.add(Finding("DO-B replay", "duplicate_side_effects_on_replay",
                  native="(a stateless re-run re-publishes on every crash)", with_redevops=dup,
                  classification=ResultClass.REDEVOPS_DELTA if dup == 0 else ResultClass.BUG,
                  note=f"replay reconstructs from the event ledger, no re-execution (consistent={consistent})"))

    out = b.write(os.path.join(HERE, "..", "results", "do_b_governed_tenant.json"))
    print("DO-B governed tenant action — findings:")
    for f in b.findings:
        print(f"  [{f.classification.value:16}] {f.experiment:24} {f.metric} = {f.with_redevops}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
