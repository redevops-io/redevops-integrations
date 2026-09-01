"""NV-B — GDPR structural closure (frameworks plan 6.4 / hyperscaler 5.4).

A regulatory question ("what must a reviewer consider when editing this GDPR article?") is answered by a NeMo
agent, but the *evidence closure* it should reason over is recovered by ReDevOps Closure Resolution. GDPR is a
structural domain: cross-referenced articles are not textually similar, so content retrieval alone is weak and the
graph is decisive. We measure seed / content / graph / composed closure recall (equal budget), reusing the frozen
thirddomain GDPR benchmark. No LLM calls — this isolates the retrieval value ReDevOps adds beneath the agent.

    python examples/gdpr_structural.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "workloads", "thirddomain"))

import statistics as st  # noqa: E402

import common as td  # noqa: E402  (thirddomain closure harness)
import financial  # noqa: E402      (GDPR cross-reference domain)
from adapters import NAT_VERSION  # noqa: E402
from bundle import AcceptanceBundle  # noqa: E402
from classification import Finding, ResultClass, classify  # noqa: E402

BUDGET = 8000                                              # matched budget where the gold closure fits (per Exp 1)


def main():
    cases = financial.cases()
    # per-case content-only vs composed (content + structural) closure recall at equal budget
    seed_only, content_only, composed = [], [], []
    for c in cases:
        tok = {u.id: u.tokens for u in c.units}
        content = td._content_scores(c)
        struct = td._struct_scores(c)
        gold = c.gold
        seed_only.append(len(set(c.seed) & gold) / len(gold))
        c_ord = sorted(content, key=lambda u: -content[u])
        content_only.append(len(td._fill(c_ord, tok, BUDGET, c.seed) & gold) / len(gold))
        tot = {u.id: content.get(u.id, 0.0) + 1.0 * struct.get(u.id, 0.0) for u in c.units}
        comp_ord = sorted(tot, key=lambda u: (-tot[u], u))
        composed.append(len(td._fill(comp_ord, tok, BUDGET, c.seed) & gold) / len(gold))

    so, co, cp = st.mean(seed_only), st.mean(content_only), st.mean(composed)
    b = AcceptanceBundle(framework="nvidia-nemo-agent-toolkit", framework_version=NAT_VERSION,
                         dataset_hashes={"gdpr": "GDPRtEXT-cross-reference-graph"},
                         config={"domain": "gdpr", "budget": BUDGET, "cases": len(cases)})

    b.add(Finding("NV-B gdpr closure-principle", "closure_recall", native=round(so, 3), with_redevops=round(cp, 3),
                  classification=classify(so, cp), note="seed-only vs composed closure — dependency-complete context"))
    b.add(Finding("NV-B gdpr structural-vs-content", "closure_recall", native=round(co, 3),
                  with_redevops=round(cp, 3), classification=classify(co, cp),
                  note="content retrieval alone is weak on a structural domain; the graph recovers the closure"))

    out = b.write(os.path.join(HERE, "..", "results", "nv_b_gdpr_structural.json"))
    print(f"NV-B GDPR structural closure — {len(cases)} articles, budget {BUDGET}")
    print(f"  seed-only {so:.3f}  |  content-only {co:.3f}  |  composed (content+graph) {cp:.3f}")
    for f in b.findings:
        print(f"  [{f.classification.value:16}] {f.experiment}: {f.native} → {f.with_redevops}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
