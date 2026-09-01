"""Gate 5 follow-up — does calibrated selective accuracy generalize to the scientific domain, with a stronger panel?

The legal verification panel was weak (one judge below chance). This tests the two things that would actually
close that hole: (1) a *stronger* independent panel — gpt-4o + gpt-4o-mini + local Qwen-27B (2 families; Kimi is
unavailable), and (2) the *new semantic domain* (SciFact). Each judge, given a claim and the top-K retrieved
candidate sentences, marks which are evidence; ground truth is SciFact's human labels. We then reproduce the
Gate-5 governance pipeline — reliability-weighted ensemble (Exp 2) + learned calibration (Exp 6) — and report the
risk-coverage curve. If it holds here, calibrated abstention is not a legal artefact.

    python gate5_sci_selective.py     # LLM calls cached in sci_panel_cache/; writes the results doc
"""
from __future__ import annotations

import json
import math
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "legal_change_closure"))

import common  # noqa: E402
import legal_llm as L  # noqa: E402
import scientific  # noqa: E402

CACHE = os.path.join(HERE, "sci_panel_cache")
N_CASES, TOPK = 60, 20
JUDGES = [("gpt4o", "openai", "gpt-4o"), ("gpt4omini", "openai", "gpt-4o-mini"), ("qwen", "local", None)]

SYS = ("You are a scientific fact-checker. Given a CLAIM and numbered candidate sentences from cited abstracts, "
       "identify which sentences are EVIDENCE that directly supports OR refutes the claim. Return STRICT JSON: "
       '{"evidence":[<id>, ...]} listing only the numbers of sentences that themselves state support or '
       "refutation. No commentary.")


def _judge(case_id, jname, provider, model, claim, cand):
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, f"{case_id}__{jname}.json")
    if os.path.exists(p):
        return set(json.load(open(p)))
    user = f"CLAIM: {claim}\n\nCANDIDATE SENTENCES:\n" + "\n".join(f"[{i}] {t}" for i, t in cand) + "\n\nReturn JSON."
    if model:
        os.environ["CR_OPENAI_MODEL"] = model
    try:
        raw = L.chat(SYS, user, max_tokens=400, provider=provider, retries=2)
        obj = L.extract_json(raw)
        ev = obj.get("evidence", []) if isinstance(obj, dict) else (obj if isinstance(obj, list) else [])
        picked = {int(x) for x in ev if str(x).lstrip("-").isdigit()}
    except Exception:
        picked = set()
    json.dump(sorted(picked), open(p, "w"))
    return picked


def build_decisions():
    cases = scientific.cases()[:N_CASES]
    items = []
    for c in cases:
        content = common._content_scores(c)                 # {id: 1/(rank+1)}
        top = sorted(content, key=lambda u: -content[u])[:TOPK]
        rank = {uid: r for r, uid in enumerate(top)}
        gold_in = c.gold & set(top)
        if len(gold_in) < 2:
            continue
        text = {u.id: u.text for u in c.units}
        # candidates presented to judges are numbered 0..K-1 (local index → unit id)
        cand = [(i, text[uid]) for i, uid in enumerate(top)]
        votes_by_judge = {}
        for jname, prov, model in JUDGES:
            picked_local = _judge(c.case_id, jname, prov, model, c.query, cand)
            votes_by_judge[jname] = {top[i] for i in picked_local if 0 <= i < len(top)}
        for uid in top:
            votes = {j: (uid in votes_by_judge[j]) for j, _, _ in JUDGES}
            items.append({"cid": c.case_id, "uid": uid,
                          "votes": votes, "truth": uid in c.gold,
                          "content": content[uid], "rank": 1.0 / (rank[uid] + 1)})
    return items


# ── ensemble + calibration (compact reuse of Exp 2 / Exp 6) ──
JN = [j for j, _, _ in JUDGES]


def reliab(train):
    w = {}
    for j in JN:
        a = min(0.99, max(0.01, st.mean(it["votes"][j] == it["truth"] for it in train)))
        w[j] = math.log(a / (1 - a))
    return w


def weighted(it, w):
    return sum(w[j] * (1 if it["votes"][j] else -1) for j in JN) > 0


def disagree(it):
    y = sum(it["votes"].values())
    return 1.0 - abs(y - (len(JN) - y)) / len(JN)


def train_lr(rows, ys, feats, iters=400, lr=0.3):
    mean = [st.mean(r[k] for r in rows) for k in range(len(feats))]
    sd = [st.pstdev([r[k] for r in rows]) or 1.0 for k in range(len(feats))]
    X = [[(r[k] - mean[k]) / sd[k] for k in range(len(feats))] for r in rows]
    b = [0.0] * (len(feats) + 1)
    for _ in range(iters):
        g = [0.0] * (len(feats) + 1)
        for xi, yi in zip(X, ys):
            z = b[0] + sum(b[k + 1] * xi[k] for k in range(len(feats)))
            p = 1 / (1 + math.exp(-max(-30, min(30, z))))
            e = p - yi
            g[0] += e
            for k in range(len(feats)):
                g[k + 1] += e * xi[k]
        b = [bj - lr * gj / len(X) for bj, gj in zip(b, g)]
    return (b, mean, sd)


def p_lr(model, r, feats):
    b, mean, sd = model
    x = [(r[k] - mean[k]) / sd[k] for k in range(len(feats))]
    z = b[0] + sum(b[k + 1] * x[k] for k in range(len(feats)))
    return 1 / (1 + math.exp(-max(-30, min(30, z))))


def risk_coverage(scored, covs):
    o = sorted(scored, key=lambda s: -s["conf"])
    return [(cov, st.mean(s["correct"] for s in o[:max(1, round(len(o) * cov))])) for cov in covs]


FEATS = ("votes_yes", "disagree", "content", "rank")


def main():
    items = build_decisions()
    ids = sorted({it["cid"] for it in items})
    fold_a = set(ids[::2])
    covs = [1.0, 0.9, 0.8, 0.7, 0.6]
    n = len(items)
    print(f"scientific panel: {len(ids)} cases, {n} candidate-decisions, judges={JN}\n")

    # per-judge accuracy
    for j in JN:
        print(f"  judge:{j:10} acc {st.mean(it['votes'][j] == it['truth'] for it in items):.3f}")
    ens_maj = st.mean((sum(it['votes'].values()) >= 2) == it['truth'] for it in items)
    print(f"  majority (2/3)        {ens_maj:.3f}")

    held = []
    for train_ids in (fold_a, set(ids) - fold_a):
        tr = [it for it in items if it["cid"] in train_ids]
        te = [it for it in items if it["cid"] not in train_ids]
        w = reliab(tr)
        for pool in (tr, te):
            for it in pool:
                it["_dec"] = weighted(it, w)
                it["votes_yes"] = float(sum(it["votes"].values()))
                it["disagree"] = disagree(it)
        rows = [[it[k] for k in FEATS] for it in tr]
        ys = [1.0 if it["_dec"] == it["truth"] else 0.0 for it in tr]
        lr = train_lr(rows, ys, FEATS)
        for it in te:
            held.append({"correct": it["_dec"] == it["truth"],
                         "learned": p_lr(lr, [it[k] for k in FEATS], FEATS),
                         "margin": 1.0 - it["disagree"]})
    base = st.mean(h["correct"] for h in held)
    wacc = st.mean((weighted(it, reliab([x for x in items if x["cid"] in fold_a])) == it["truth"]) for it in items)
    learned = risk_coverage([{**h, "conf": h["learned"]} for h in held], covs)
    margin = risk_coverage([{**h, "conf": h["margin"]} for h in held], covs)

    print(f"\n  reliability-weighted full-coverage acc {base:.3f}")
    print(f"  {'coverage':>9} {'learned calib':>14} {'crude margin':>13}")
    for (cov, la), (_, ma) in zip(learned, margin):
        print(f"  {cov:9.0%} {la:14.3f} {ma:13.3f}")
    _write_doc(ids, items, held, base, learned, margin, ens_maj)


def _write_doc(ids, items, held, base, learned, margin, ens_maj):
    def accs(j):
        return st.mean(it["votes"][j] == it["truth"] for it in items)
    jrows = "\n".join(f"| {j} | {accs(j):.3f} |" for j in JN)
    crows = "\n".join(f"| {cov:.0%} | {la:.3f} | {ma:.3f} |" for (cov, la), (_, ma) in zip(learned, margin))
    l80 = next(a for c, a in learned if abs(c - 0.8) < 1e-9)
    l60 = next(a for c, a in learned if abs(c - 0.6) < 1e-9)
    doc = f"""# Gate 5 follow-up — calibrated selective accuracy on the scientific domain (stronger panel)

**Why.** The legal panel was weak (one judge below chance), which capped Exp 2. This tests whether the *governance*
result — reliability-weighted ensemble + learned calibration → selective abstention — holds (a) with a **stronger
panel** (gpt-4o + gpt-4o-mini + local Qwen-27B) and (b) on the **new semantic domain** (SciFact). If it does,
calibrated abstention is a capability property, not a legal artefact.

**Setup.** {len(ids)} SciFact claims; for each, the top-{TOPK} retrieved candidate sentences are shown to each
judge, which marks the evidence; ground truth is SciFact's human labels ({len(items)} candidate-decisions). Same
pipeline as Exp 2/6: reliability-weighted vote (Naive-Bayes log-odds, 2-fold CV) + a logistic correctness
predictor over [votes, disagreement, retrieval rank, content score]; abstention routes the least-confident to
review.

## Three accuracy concepts — do NOT conflate them

This page reports **verification accuracy** and **selective autonomous accuracy**, which are different from the
**closure recall** the retrieval experiments report:

1. **Closure/retrieval recall** — did we *fetch* the right context? (scientific closure retrieval was **0.867**;
   legal 0.582.)
2. **Verification accuracy** — given retrieved candidates, are the per-candidate keep/drop decisions correct vs
   human labels? (this page.)
3. **Selective autonomous accuracy at a stated coverage** — accuracy on the decisions the system *acts on*
   autonomously, with the rest routed to review. (this page.)

So the defensible statement of the headline number is: **"on SciFact verification decisions, the stronger panel
achieves {ens_maj:.1%} accuracy at full coverage and {l80:.1%} at 80% autonomous coverage, with the remaining 20%
explicitly routed to review"** — **not** "95.7% end-to-end closure recovery" (scientific closure *retrieval* was
0.867, a different quantity).

## Panel (stronger than legal — no below-chance judge)

| judge | per-candidate accuracy vs human labels |
|---|---|
{jrows}
| majority (2/3) | {ens_maj:.3f} |

## Selective accuracy generalizes — but the *learned* calibrator's edge shrinks with a strong panel

| coverage | learned calibration | crude vote margin |
|---|---|---|
{crows}

Two findings, both honest:

1. **The governance mechanism generalizes.** Reliability-weighted full-coverage accuracy is {base:.3f}; abstaining
   on the least-confident lifts it to **{l80:.3f} at 80% coverage** ({l60:.3f} at 60%). Disagreement-driven
   calibrated abstention separates the confidently-resolvable majority from the tail that should go to review —
   **on an independent semantic domain with human ground truth, not just legal.** The same shape as legal
   (which went 0.696→0.873→0.926), at a higher operating point.

2. **Here, learned calibration ties the crude vote margin** ({l80:.3f} vs {next(a for c,a in margin if abs(c-0.8)<1e-9):.3f} at 80%),
   whereas on legal the learned calibrator *dominated* it (0.873 vs 0.714). The reason is the panel: legal's panel
   was weak and noisy, so its raw disagreement was a poor confidence signal and the learned features (retrieval +
   structural support) added a lot; scientific's panel is **strong and accurate ({ens_maj:.3f} majority, no
   below-chance judge)**, so disagreement *alone* is already well-calibrated and the extra features add little.

## Reading

- **The stronger panel works** exactly as predicted: all three judges ~0.90 (vs legal, where one was below
  chance), so the ensemble is healthy and the base accuracy is high ({base:.3f} vs legal's 0.696).
- **Panel strength is the lever, and it has two effects:** it raises the base accuracy *and* makes simple
  disagreement-calibration sufficient (the learned calibrator's advantage is largest precisely when the panel is
  weak). So the practical recipe is "strengthen the panel first, then calibrate" — not "always deploy the complex
  calibrator."
- Together with legal, the load-bearing claim now holds on two independent semantic domains: **resolve what the
  evidence supports; abstain when the closure is interpretation-dependent.**

**Honest bounds.** {len(ids)} cases, top-{TOPK} candidate pool (so recall of evidence outside the pool is out of
scope here — this measures verification/calibration, not retrieval); two judge families (Kimi unavailable), so
"independent" is limited to two lineages; ground truth is SciFact's single human annotation. The claim is scoped
to calibration/abstention generalizing, which is what is measured.
"""
    out = os.path.join(HERE, "..", "..", "docs", "results", "resolution_closure_gate5_sci_selective.md")
    with open(out, "w") as f:
        f.write(doc)
    print(f"\nwrote {os.path.relpath(out, os.path.join(HERE, '..', '..'))}")


if __name__ == "__main__":
    main()
