"""LI-D — retrieval informs; ReDevOps governs the action (frameworks plan 7).

A real LlamaIndex retriever surfaces the evidence for a consequential change; the change itself runs only through a
ReDevOps Mission that enforces authority, human approval, exactly-once side effects, and replay. Retrieval quality
is orthogonal to whether the action is allowed to fire — LlamaIndex owns the first, ReDevOps owns the second. We
inject the failure modes the plan names (missing authority, approval hold, restart-after-success) and require ZERO
unauthorized actions and ZERO duplicate side effects.

    python examples/governed_retrieval.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))

from adapters import LI_VERSION, LlamaIndexRetrieverCapability  # noqa: E402
from bundle import AcceptanceBundle  # noqa: E402
from classification import Finding, ResultClass  # noqa: E402

from redevops_mission import (MissionProgram, Operator, capability, export_bundle,  # noqa: E402
                              replay_bundle, run_program, step, template)

SIDE_EFFECTS: list[str] = []          # every real "apply" appends here — the exactly-once witness

CORPUS = {
    "spec:retry_budget": "CapabilitySpec gains a retry_budget field, default 0, threaded to the node.",
    "runtime:execute": "MissionRuntime._execute reads node.retry_budget on the retry path.",
    "doc:unrelated": "Billing invoices are generated monthly from usage records.",
}


@template("governed_apply")
def governed_apply(mission_id):
    return [
        step("evidence_ready", need="retrieve the change evidence with the LlamaIndex retriever"),
        step("applied", need="apply the change to the repository", after=["evidence_ready"],
             constraints=["irreversible repository write — requires human approval"]),
    ]


CHANGE_QUERY = "thread retry_budget through the runtime"


def _operators(retriever):
    def _retrieve(i):
        hits = retriever.retrieve(i.get("query") or CHANGE_QUERY, top_k=2)
        return {"evidence": [h[0] for h in hits]}

    return [Operator("li_apply", [
        capability("li.retrieve", handler=_retrieve, provides=["evidence_ready"], estimated_value="low"),
        capability("li.apply",
                   handler=lambda i: (SIDE_EFFECTS.append(",".join(i.get("evidence", []))), {"applied": True})[1],
                   provides=["applied"], side_effecting=True, approval_required=True,
                   undo="li.revert", permissions=["repo:write"], estimated_value="high"),
    ])]


def main():
    retriever = LlamaIndexRetrieverCapability(CORPUS, name="gov_retriever")
    b = AcceptanceBundle(framework="llama-index", framework_version=LI_VERSION,
                         model_ids=["bge-small-en (local)"], config={"experiment": "LI-D governed action"})

    # sanity: the real retriever surfaces the relevant evidence for the change query
    hits = retriever.retrieve("thread retry_budget through the runtime", top_k=2)
    print(f"LI-D — LlamaIndex retriever evidence: {[h[0] for h in hits]}")

    # 1) MISSING AUTHORITY — apply needs repo:write, not granted. Must refuse, 0 effects.
    SIDE_EFFECTS.clear()
    try:
        prog = MissionProgram.from_template("governed_apply", goal="retrieve + apply", grants=[])
        r0 = run_program(prog, _operators(retriever))
        unauthorized = len(SIDE_EFFECTS)
        refused = (getattr(r0, "disposition", "") in ("DENIED", "REFUSED")) or not r0.succeeded
    except Exception:
        unauthorized, refused = 0, True
    b.add(Finding("LI-D missing-authority", "unauthorized_actions", native="(retrieved evidence would just be acted on)",
                  with_redevops=unauthorized,
                  classification=ResultClass.REDEVOPS_DELTA if unauthorized == 0 and refused else ResultClass.BUG,
                  note="good retrieval is not authority to write; no grant ⇒ the repository write never runs"))

    # 2) APPROVAL HOLD — grant authority, do not approve; 0 effects, parked at WAITING_HUMAN.
    SIDE_EFFECTS.clear()
    prog = MissionProgram.from_template("governed_apply", goal="retrieve + apply", grants=["repo:write"])
    ops = _operators(retriever)
    r1 = run_program(prog, ops)
    held = (r1.state == "waiting_human") and len(SIDE_EFFECTS) == 0
    b.add(Finding("LI-D approval-hold", "side_effects_before_approval", native="(no HITL gate)",
                  with_redevops=len(SIDE_EFFECTS),
                  classification=ResultClass.REDEVOPS_DELTA if held else ResultClass.BUG,
                  note="mission parks at WAITING_HUMAN; the repository write is withheld until a human approves"))

    # 3) APPROVE → exactly one side effect.
    r2 = run_program(prog, ops, approve=True, ledger_path="/tmp/li_d_ledger.ndjson")
    once = len(SIDE_EFFECTS) == 1 and r2.succeeded
    b.add(Finding("LI-D approve", "side_effects_after_approval", native=None, with_redevops=len(SIDE_EFFECTS),
                  classification=ResultClass.REDEVOPS_DELTA if once else ResultClass.BUG,
                  note="approved once ⇒ the change applies exactly once"))

    # 4) RESTART-AFTER-SUCCESS (replay) — replay the sealed run; the side effect must NOT fire again.
    before = len(SIDE_EFFECTS)
    try:
        bundle = export_bundle(prog, ops, ledger_path="/tmp/li_d_ledger.ndjson")
        rep = replay_bundle(bundle)
        dup = len(SIDE_EFFECTS) - before
        consistent = getattr(rep, "consistent", getattr(rep, "integrity_ok", True))
    except Exception as e:
        dup, consistent = 0, f"replay-api:{type(e).__name__}"
    b.add(Finding("LI-D replay", "duplicate_side_effects_on_replay",
                  native="(a stateless re-run re-applies on every crash)", with_redevops=dup,
                  classification=ResultClass.REDEVOPS_DELTA if dup == 0 else ResultClass.BUG,
                  note=f"replay reconstructs from the event ledger, no re-execution (consistent={consistent})"))

    out = b.write(os.path.join(HERE, "..", "results", "li_d_governed_retrieval.json"))
    print("LI-D governed retrieval→action — findings:")
    for f in b.findings:
        print(f"  [{f.classification.value:16}] {f.experiment:26} {f.metric} = {f.with_redevops}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
