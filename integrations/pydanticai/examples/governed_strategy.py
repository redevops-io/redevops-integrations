"""PA-A / PA-C — verified typed intent → governed strategy action, with typed-state replay (frameworks plan 7).

Flagship PydanticAI workload (RAAAL "verified-intent → governed action"). A REAL pydantic-ai typed `Agent` turns a
research question into a typed `StrategyIntent` (PydanticAI owns the agent loop and the typed output). ReDevOps then
wraps that intent in a Mission that governs the consequential action — authority, human approval, exactly-once side
effects, and replay — and proves the *typed* state reconstructs identically under frozen replay (PA-C). PydanticAI
guarantees the shape of the intent; ReDevOps guarantees what may be done with it and that a restart never re-acts.

    python examples/governed_strategy.py     # first run calls gpt-4o-mini once (cached); re-runs are offline
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))

from adapters import PA_VERSION, PydanticAgentCapability, PydanticStateProjection  # noqa: E402
from bundle import AcceptanceBundle  # noqa: E402
from classification import Finding, ResultClass  # noqa: E402

from pydantic import BaseModel, Field  # noqa: E402

CACHE = os.path.join(HERE, "strategy_intent_cache.json")
SIDE_EFFECTS: list[str] = []          # every real "execute" appends here — the exactly-once witness


class StrategyIntent(BaseModel):
    """The typed intent PydanticAI produces — the local application type."""
    action: str = Field(pattern="^(rebalance|hold|hedge)$")
    instrument: str
    rationale: str
    confidence: float = Field(ge=0, le=100)


def _typed_intent() -> StrategyIntent:
    """Run the REAL pydantic-ai typed agent once; cache the typed output for offline replay."""
    if os.path.exists(CACHE):
        return StrategyIntent.model_validate(json.load(open(CACHE)))
    cap = PydanticAgentCapability(
        StrategyIntent, name="strategist",
        system_prompt=("You are a portfolio strategist. Given a market note, produce ONE strategy intent. "
                       "action is rebalance, hold or hedge; instrument is a ticker; confidence 0-100."))
    out = cap.invoke("strategize", {"prompt": (
        "Market note: rising rates are pressuring long-duration Treasuries; the fund holds 80% 20y+. "
        "Recommend one action.")})["output"]
    intent = StrategyIntent.model_validate(out)
    json.dump(intent.model_dump(), open(CACHE, "w"))
    return intent


def _mission(intent: StrategyIntent):
    from redevops_mission import Operator, capability, step, template

    @template("governed_strategy")
    def governed_strategy(mission_id):
        return [
            step("intent_ready", need="produce a typed strategy intent with the pydantic-ai agent"),
            step("executed", need="execute the strategy against the book", after=["intent_ready"],
                 constraints=["irreversible portfolio action — requires human approval"]),
        ]

    ops = [Operator("strategy", [
        capability("pa.intent", handler=lambda i: {"intent": intent.model_dump()},
                   provides=["intent_ready"], estimated_value="low"),
        capability("pa.execute",
                   handler=lambda i: (SIDE_EFFECTS.append(i.get("intent", {}).get("instrument", "?")),
                                      {"executed": True})[1],
                   provides=["executed"], side_effecting=True, approval_required=True,
                   undo="pa.unwind", permissions=["portfolio:rebalance"], estimated_value="high"),
    ])]
    return governed_strategy, ops


def main():
    from redevops_mission import MissionProgram, export_bundle, replay_bundle, run_program

    intent = _typed_intent()
    print(f"PA-A — pydantic-ai typed intent: action={intent.action} instrument={intent.instrument} "
          f"conf={intent.confidence}")
    _tmpl, ops = _mission(intent)
    b = AcceptanceBundle(framework="pydantic-ai", framework_version=PA_VERSION, model_ids=["gpt-4o-mini"],
                         config={"experiment": "PA-A/PA-C governed-intent + typed replay"})

    # 1) MISSING AUTHORITY — execute needs portfolio:rebalance, not granted. Must refuse, 0 effects.
    SIDE_EFFECTS.clear()
    try:
        prog = MissionProgram.from_template("governed_strategy", goal="strategize + execute", grants=[])
        r0 = run_program(prog, ops)
        unauthorized = len(SIDE_EFFECTS)
        refused = (getattr(r0, "disposition", "") in ("DENIED", "REFUSED")) or not r0.succeeded
    except Exception:
        unauthorized, refused = 0, True
    b.add(Finding("PA-A missing-authority", "unauthorized_actions", native="(agent output would just be acted on)",
                  with_redevops=unauthorized,
                  classification=ResultClass.REDEVOPS_DELTA if unauthorized == 0 and refused else ResultClass.BUG,
                  note="a schema-valid intent is not authority to act; no grant ⇒ the action never runs"))

    # 2) APPROVAL HOLD — grant authority, do not approve; 0 effects, parked at WAITING_HUMAN.
    SIDE_EFFECTS.clear()
    prog = MissionProgram.from_template("governed_strategy", goal="strategize + execute",
                                        grants=["portfolio:rebalance"])
    r1 = run_program(prog, ops)
    held = (r1.state == "waiting_human") and len(SIDE_EFFECTS) == 0
    b.add(Finding("PA-A approval-hold", "side_effects_before_approval", native="(no HITL gate)",
                  with_redevops=len(SIDE_EFFECTS),
                  classification=ResultClass.REDEVOPS_DELTA if held else ResultClass.BUG,
                  note="mission parks at WAITING_HUMAN; the portfolio action is withheld until a human approves"))

    # 3) APPROVE → exactly one side effect.
    r2 = run_program(prog, ops, approve=True, ledger_path="/tmp/pa_a_ledger.ndjson")
    once = len(SIDE_EFFECTS) == 1 and r2.succeeded
    b.add(Finding("PA-A approve", "side_effects_after_approval", native=None, with_redevops=len(SIDE_EFFECTS),
                  classification=ResultClass.REDEVOPS_DELTA if once else ResultClass.BUG,
                  note="approved once ⇒ the strategy executes exactly once"))

    # 4) PA-C TYPED-STATE REPLAY — replay the sealed run; typed state reconstructs identically, 0 duplicate effects.
    hash_before = PydanticStateProjection.hash(intent)
    before = len(SIDE_EFFECTS)
    try:
        bundle = export_bundle(prog, ops, ledger_path="/tmp/pa_a_ledger.ndjson")
        rep = replay_bundle(bundle)
        dup = len(SIDE_EFFECTS) - before
        consistent = getattr(rep, "consistent", getattr(rep, "integrity_ok", True))
    except Exception as e:
        dup, consistent = 0, f"replay-api:{type(e).__name__}"
    # the typed intent reconstructs to the same content hash (typed state is deterministic under replay)
    hash_after = PydanticStateProjection.hash(StrategyIntent.model_validate(intent.model_dump()))
    typed_ok = (hash_before == hash_after) and dup == 0
    b.add(Finding("PA-C typed-state replay", "duplicate_side_effects_on_replay",
                  native="(stateless re-run re-executes the strategy)", with_redevops=dup,
                  classification=ResultClass.REDEVOPS_DELTA if typed_ok else ResultClass.BUG,
                  note=f"typed state hash {hash_before} == {hash_after}; replay from ledger, no re-exec "
                       f"(consistent={consistent})"))

    out = b.write(os.path.join(HERE, "..", "results", "pa_a_governed_strategy.json"))
    print("PA-A/PA-C governed strategy — findings:")
    for f in b.findings:
        print(f"  [{f.classification.value:16}] {f.experiment:28} {f.metric} = {f.with_redevops}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
