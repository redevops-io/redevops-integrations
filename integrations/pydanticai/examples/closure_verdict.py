"""PA-E — Closure Resolution context under a typed pydantic-ai verdict agent (frameworks plan 7).

A REAL pydantic-ai typed `Agent` returns a typed `Verdict` for scientific claims. We give the SAME agent (a) naive
semantic context and (b) ReDevOps Closure-Resolution context at equal budget/model, and measure typed-verdict
accuracy, closure recall, and selective autonomy (the two context views act as a cheap ensemble: agree ⇒ resolve
autonomously, disagree ⇒ REQUIRE_REVIEW). Pydantic guarantees the verdict is well-typed; Closure Resolution changes
whether it is *correct*. Real pydantic-ai agent; gpt-4o-mini; typed outputs cached for offline replay.

    python examples/closure_verdict.py
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "workloads", "thirddomain"))

import common as td  # noqa: E402
import scientific  # noqa: E402
from adapters import PA_VERSION, PydanticAgentCapability  # noqa: E402
from bundle import AcceptanceBundle  # noqa: E402
from classification import Finding, ResultClass, classify  # noqa: E402

from pydantic import BaseModel, Field  # noqa: E402

BUDGET = 300
N_PER_LABEL = 15
CACHE = os.path.join(HERE, "closure_verdict_cache")
VERDICT = {"SUPPORT": "Supported", "CONTRADICT": "Refuted"}


class Verdict(BaseModel):
    """Typed verdict — the type guarantees the shape; structure alone doesn't establish correctness."""
    label: str = Field(pattern="^(Supported|Refuted|NotEnoughInfo)$")


_agent = PydanticAgentCapability(
    Verdict, name="scifact_verifier",
    system_prompt=("You are a scientific fact-checker. Given a CLAIM and EVIDENCE sentences, decide if the evidence "
                   "Supports the claim, Refutes it, or gives NotEnoughInfo. Return the label only."))


def _verdict_map():
    root = os.path.join(HERE, "..", "..", "..", "workloads", "thirddomain", "data", "scifact", "data")
    m = {}
    for split in ("claims_train", "claims_dev"):
        p = os.path.join(root, f"{split}.jsonl")
        if not os.path.exists(p):
            continue
        for line in open(p):
            c = json.loads(line); ev = c.get("evidence") or {}
            if not ev:
                continue
            labs = {g["label"] for gs in ev.values() for g in gs}
            v = "SUPPORT" if "SUPPORT" in labs else ("CONTRADICT" if "CONTRADICT" in labs else None)
            if v:
                m[str(c["id"])] = VERDICT[v]
    return m


def _context(case, ids_by_arm):
    text = {u.id: u.text for u in case.units}
    return {arm: "\n".join(f"- {text[i]}" for i in ids) for arm, ids in ids_by_arm.items()}


def _ask(case_id, arm, claim, ctx):
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, f"{case_id}__{arm}.json")
    if os.path.exists(p):
        return json.load(open(p))
    out = _agent.invoke("verify", {"prompt": f"CLAIM: {claim}\n\nEVIDENCE:\n{ctx}\n\nLabel:"})["output"]
    v = out.get("label", "NotEnoughInfo")
    json.dump(v, open(p, "w"))
    return v


def main():
    vmap = _verdict_map()
    cases = [c for c in scientific.cases() if c.case_id.split("-")[-1] in vmap]
    picked = {"Supported": [], "Refuted": []}
    for c in cases:
        v = vmap[c.case_id.split("-")[-1]]
        if len(picked[v]) < N_PER_LABEL:
            picked[v].append((c, v))
    sample = picked["Supported"] + picked["Refuted"]
    print(f"PA-E SciFact — {len(sample)} claims (real pydantic-ai typed agent, gpt-4o-mini)")

    naive_ok, closure_ok, recalls_n, recalls_c, agree = [], [], [], [], []
    for c, gold in sample:
        tok = {u.id: u.tokens for u in c.units}
        content = td._content_scores(c); struct = td._struct_scores(c)
        naive_ids = td._fill(sorted(content, key=lambda u: -content[u]), tok, BUDGET, c.seed)
        tot = {u.id: content.get(u.id, 0.0) + 0.3 * struct.get(u.id, 0.0) for u in c.units}
        closure_ids = td._fill(sorted(tot, key=lambda u: (-tot[u], u)), tok, BUDGET, c.seed)
        recalls_n.append(len(naive_ids & c.gold) / len(c.gold))
        recalls_c.append(len(closure_ids & c.gold) / len(c.gold))
        ctx = _context(c, {"naive": sorted(naive_ids), "closure": sorted(closure_ids)})
        vn = _ask(c.case_id, "naive", c.query, ctx["naive"])
        vc = _ask(c.case_id, "closure", c.query, ctx["closure"])
        naive_ok.append(vn == gold); closure_ok.append(vc == gold); agree.append(vn == vc)

    na, ca = st.mean(naive_ok), st.mean(closure_ok)
    rn, rc = st.mean(recalls_n), st.mean(recalls_c)
    auto = [ok for ok, ag in zip(closure_ok, agree) if ag]
    cov = len(auto) / len(sample)
    routed = [ok for ok, ag in zip(closure_ok, agree) if not ag]

    b = AcceptanceBundle(framework="pydantic-ai", framework_version=PA_VERSION, model_ids=["gpt-4o-mini"],
                         dataset_hashes={"scifact": "allenai/scifact"},
                         config={"domain": "scifact", "budget": BUDGET, "n": len(sample)})
    b.add(Finding("PA-E verification accuracy", "task_accuracy", native=round(na, 3), with_redevops=round(ca, 3),
                  classification=classify(na, ca),
                  note="typed pydantic-ai verdict on naive vs closure context (equal budget/model)"))
    b.add(Finding("PA-E evidence recall", "closure_recall", native=round(rn, 3), with_redevops=round(rc, 3),
                  classification=classify(rn, rc), note="gold evidence sentences recovered"))
    b.add(Finding("PA-E selective autonomy", "autonomous_accuracy@coverage",
                  native="(a typed verdict carries no abstention signal)", with_redevops=f"{st.mean(auto):.3f} @ {cov:.0%}",
                  classification=ResultClass.REDEVOPS_DELTA,
                  note=f"agree⇒resolve, disagree⇒REQUIRE_REVIEW; routed-tail acc {st.mean(routed) if routed else 0:.3f}"))

    out = b.write(os.path.join(HERE, "..", "results", "pa_e_closure_verdict.json"))
    print(f"  verification accuracy: naive {na:.3f} → closure {ca:.3f}")
    print(f"  evidence recall:       naive {rn:.3f} → closure {rc:.3f}")
    print(f"  selective autonomy:    {st.mean(auto):.3f} @ {cov:.0%} coverage  (routed tail "
          f"{st.mean(routed) if routed else 0:.3f})")
    for f in b.findings:
        print(f"  [{f.classification.value:16}] {f.experiment}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
