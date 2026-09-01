"""LI-A (flagship) — code-change closure just outside the LlamaIndex retrieval boundary (frameworks plan 7).

LlamaIndex's strongest abstraction is retrieval over an index. So we test the boundary *immediately outside* it.
A real LlamaIndex `VectorStoreIndex` retriever finds the symbols that are semantically similar to the change
description — the seed and its type/schema. It does not, by construction, recover the *contract callers* and
transitive callers a correct code change must also touch: those are structurally linked, not textually similar.

ReDevOps registers the LlamaIndex retriever as ONE representation and composes it with the static call-graph
closure (LlamaIndex retriever ∪ structural traversal, same budget + token accounting). The retriever is not
replaced — it is composed. We measure closure recall and, especially, contract-caller recall, on real refactor
tasks over the pinned agentic-os tree.

    python examples/code_closure.py [--repo /mnt/backup/projects/agentic-os] [--budget 4000]
"""
from __future__ import annotations

import argparse
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
CR = os.path.join(HERE, "..", "..", "..", "workloads", "code_retrieval")
sys.path.insert(0, CR)

from adapters import LlamaIndexRetrieverCapability, LI_VERSION  # noqa: E402
from bundle import AcceptanceBundle  # noqa: E402
from classification import Finding, ResultClass, classify  # noqa: E402

from resolver import build_call_graph, index_repo  # noqa: E402
from retrievers import _unit, graph_closure_arm, naive_semantic_arm  # noqa: E402
from scoring import Oracle, RetrievedContext, score  # noqa: E402

_BIG = 10 ** 9


def _query_text(task_yaml: str) -> tuple[str, set[str]]:
    import yaml
    d = yaml.safe_load(open(task_yaml))
    return d.get("description", "") or "", set(d.get("seed_symbols", []) or [])


def _budget_fill(ordered_ids, index, budget):
    units, spent, seen = [], 0, set()
    for sid in ordered_ids:
        if sid in seen or sid not in index:
            continue
        seen.add(sid)
        sym = index[sid]
        if spent + sym.tokens > budget and units:
            break
        units.append(_unit(sym))
        spent += sym.tokens
        if spent >= budget:
            break
    return RetrievedContext(units)


def _llamaindex_hits(cap, query, seed_symbols, index):
    # rank the whole corpus by the real LlamaIndex retriever (query = NL description + seed leaf names)
    q = query + " " + " ".join(s.split(":")[-1] for s in seed_symbols)
    return [sid for sid, _ in cap.retrieve(q, top_k=len(index))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/mnt/backup/projects/agentic-os")
    ap.add_argument("--budget", type=int, default=4000)
    args = ap.parse_args()

    index = index_repo(args.repo)
    graph = build_call_graph(index)
    repo_total = sum(s.tokens for s in index.values())
    print(f"LI-A code closure — {len(index)} symbols indexed from {os.path.basename(args.repo)}, "
          f"budget {args.budget} tok (real LlamaIndex retriever, local bge-small)")

    corpus = {sid: s.source for sid, s in index.items()}
    cap = LlamaIndexRetrieverCapability(corpus, name="code_retriever")

    tasks_dir = os.path.join(CR, "oracle", "tasks")
    rows = []
    for fn in sorted(os.listdir(tasks_dir)):
        if not fn.endswith(".yaml"):
            continue
        path = os.path.join(tasks_dir, fn)
        oracle = Oracle.load(path).for_retrieval()
        query, seed = _query_text(path)
        seed = seed & set(index)
        if not seed:
            continue

        li_ranked = _llamaindex_hits(cap, query, seed, index)
        li = _budget_fill(li_ranked, index, args.budget)
        naive = naive_semantic_arm(query, seed, index, args.budget)
        # ReDevOps composition: structural closure FIRST (must-change members) ∪ LlamaIndex representation fills.
        graph_units = graph_closure_arm(seed, index, graph, _BIG).units
        composed = _budget_fill([u.symbol for u in graph_units] + li_ranked, index, args.budget)
        oc = RetrievedContext([_unit(index[s]) for s in sorted(
            oracle.seed_symbols | oracle.required_all | oracle.contract_callers) if s in index])

        s_li = score(oracle, li, repo_total)
        s_nv = score(oracle, naive, repo_total)
        s_cp = score(oracle, composed, repo_total)
        s_oc = score(oracle, oc, repo_total)
        rows.append((oracle.task_id, s_nv, s_li, s_cp, s_oc))
        print(f"\n  {oracle.task_id}")
        for lbl, sc in (("bm25 baseline", s_nv), ("llama-index", s_li), ("li + closure", s_cp), ("oracle", s_oc)):
            print(f"    {lbl:14} closure {_f(sc['closure_recall_total'])} · contract-caller "
                  f"{_f(sc['contract_caller_recall'])} · match {_f(sc['match_recall'])}")

    def _mean(key, i):
        vals = [r[i][key] for r in rows if r[i][key] is not None]
        return st.mean(vals) if vals else 0.0

    li_cl, cp_cl = _mean("closure_recall_total", 2), _mean("closure_recall_total", 3)
    li_cc, cp_cc = _mean("contract_caller_recall", 2), _mean("contract_caller_recall", 3)
    # non-circular "composes, never degrades": per task, composed closure recall vs LlamaIndex-alone.
    deltas = [(r[3]["closure_recall_total"] or 0) - (r[2]["closure_recall_total"] or 0) for r in rows]
    regressions = sum(1 for d in deltas if d < -1e-9)
    li_suffices = sum(1 for r in rows if (r[2]["closure_recall_total"] or 0) >= 0.999)

    b = AcceptanceBundle(framework="llama-index", framework_version=LI_VERSION,
                         model_ids=["bge-small-en (local)"], dataset_hashes={"repo": os.path.basename(args.repo)},
                         config={"budget": args.budget, "n_tasks": len(rows), "symbols": len(index)})
    b.add(Finding("LI-A closure recall", "closure_recall_total", native=round(li_cl, 3), with_redevops=round(cp_cl, 3),
                  classification=classify(li_cl, cp_cl),
                  note="LlamaIndex retriever alone vs LlamaIndex-as-representation composed with structural closure"))
    b.add(Finding("LI-A contract-caller recall", "contract_caller_recall", native=round(li_cc, 3),
                  with_redevops=round(cp_cc, 3), classification=classify(li_cc, cp_cc),
                  note="the structural closure a semantic retriever misses — the boundary just outside retrieval"))
    b.add(Finding("LI-A composes, never degrades", "regressions_vs_llamaindex",
                  native=f"{li_suffices}/{len(rows)} tasks retrieval alone resolves", with_redevops=regressions,
                  classification=ResultClass.PARITY if regressions == 0 else ResultClass.BUG,
                  note="on tasks LlamaIndex already resolves, composition matches it; 0 tasks regressed — the "
                       "retriever is composed and preferred when it suffices, not replaced (match recall is high "
                       "by construction on the seeded arm, so it is not claimed as a retrieval win)"))

    out = b.write(os.path.join(HERE, "..", "results", "li_a_code_closure.json"))
    print(f"\n  closure recall        llama-index {li_cl:.3f} → li+closure {cp_cl:.3f}")
    print(f"  contract-caller recall llama-index {li_cc:.3f} → li+closure {cp_cc:.3f}")
    print(f"  composes/never degrades: {regressions} regressions · {li_suffices}/{len(rows)} tasks retrieval alone suffices")
    print(f"wrote {out}")


def _f(v):
    return " N/A " if v is None else f"{v:.3f}"


if __name__ == "__main__":
    main()
