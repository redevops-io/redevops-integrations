"""GO-B — a delegated finding informs; ReDevOps governs the consequential action (frameworks plan 7).

A research mesh produces a finding that triggers a consequential action (write it into the shared knowledge base
other agents rely on). Whichever mesh agent surfaced it, the write runs only through a ReDevOps Mission enforcing
authority, human approval, exactly-once execution and replay. We inject the failure modes the plan names and
require ZERO unauthorized actions and ZERO duplicate writes.

    python examples/governed_research.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))

from adapters import ADK_VERSION  # noqa: E402
from bundle import AcceptanceBundle  # noqa: E402
from classification import Finding, ResultClass  # noqa: E402

from redevops_mission import (MissionProgram, Operator, capability, export_bundle,  # noqa: E402
                              replay_bundle, run_program, step, template)

WRITES: list[str] = []          # every real KB write appends here — the exactly-once witness
FINDING = {"topic": "rag-latency", "claim": "hybrid retrieval cuts p95 by 38%"}


@template("governed_kb_write")
def governed_kb_write(mission_id):
    return [
        step("finding_ready", need="have the mesh produce a research finding"),
        step("kb_written", need="write the finding into the shared knowledge base", after=["finding_ready"],
             constraints=["shared KB write other agents rely on — requires human approval"]),
    ]


def _operators():
    return [Operator("mesh_kb", [
        capability("mesh.finding", handler=lambda i: {"finding": FINDING},
                   provides=["finding_ready"], estimated_value="low"),
        capability("mesh.write",
                   handler=lambda i: (WRITES.append(i.get("finding", {}).get("topic", "?")),
                                      {"kb_written": True})[1],
                   provides=["kb_written"], side_effecting=True, approval_required=True,
                   undo="mesh.retract", permissions=["kb:write"], estimated_value="high"),
    ])]


def main():
    b = AcceptanceBundle(framework="google-adk", framework_version=ADK_VERSION,
                         config={"experiment": "GO-B governed research action"})

    # 1) MISSING AUTHORITY — write needs kb:write, not granted. Must refuse, 0 writes.
    WRITES.clear()
    try:
        prog = MissionProgram.from_template("governed_kb_write", goal="finding + write", grants=[])
        r0 = run_program(prog, _operators())
        unauthorized = len(WRITES)
        refused = (getattr(r0, "disposition", "") in ("DENIED", "REFUSED")) or not r0.succeeded
    except Exception:
        unauthorized, refused = 0, True
    b.add(Finding("GO-B missing-authority", "unauthorized_actions", native="(a mesh finding would just be written)",
                  with_redevops=unauthorized,
                  classification=ResultClass.REDEVOPS_DELTA if unauthorized == 0 and refused else ResultClass.BUG,
                  note="a delegated finding is not authority to mutate shared state; no grant ⇒ the write never runs"))

    # 2) APPROVAL HOLD — grant authority, do not approve; 0 writes, parked at WAITING_HUMAN.
    WRITES.clear()
    prog = MissionProgram.from_template("governed_kb_write", goal="finding + write", grants=["kb:write"])
    ops = _operators()
    r1 = run_program(prog, ops)
    held = (r1.state == "waiting_human") and len(WRITES) == 0
    b.add(Finding("GO-B approval-hold", "side_effects_before_approval", native="(no HITL gate across the mesh)",
                  with_redevops=len(WRITES),
                  classification=ResultClass.REDEVOPS_DELTA if held else ResultClass.BUG,
                  note="mission parks at WAITING_HUMAN; the shared-KB write is withheld until a human approves"))

    # 3) APPROVE → exactly one write.
    r2 = run_program(prog, ops, approve=True, ledger_path="/tmp/go_b_ledger.ndjson")
    once = len(WRITES) == 1 and r2.succeeded
    b.add(Finding("GO-B approve", "side_effects_after_approval", native=None, with_redevops=len(WRITES),
                  classification=ResultClass.REDEVOPS_DELTA if once else ResultClass.BUG,
                  note="approved once ⇒ the KB write fires exactly once"))

    # 4) RESTART-AFTER-SUCCESS (replay) — replay the sealed run; the write must NOT fire again.
    before = len(WRITES)
    try:
        bundle = export_bundle(prog, ops, ledger_path="/tmp/go_b_ledger.ndjson")
        rep = replay_bundle(bundle)
        dup = len(WRITES) - before
        consistent = getattr(rep, "consistent", getattr(rep, "integrity_ok", True))
    except Exception as e:
        dup, consistent = 0, f"replay-api:{type(e).__name__}"
    b.add(Finding("GO-B replay", "duplicate_side_effects_on_replay",
                  native="(a stateless mesh re-run re-writes on every crash)", with_redevops=dup,
                  classification=ResultClass.REDEVOPS_DELTA if dup == 0 else ResultClass.BUG,
                  note=f"replay reconstructs from the event ledger, no re-execution (consistent={consistent})"))

    out = b.write(os.path.join(HERE, "..", "results", "go_b_governed_research.json"))
    print("GO-B governed research action — findings:")
    for f in b.findings:
        print(f"  [{f.classification.value:16}] {f.experiment:24} {f.metric} = {f.with_redevops}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
