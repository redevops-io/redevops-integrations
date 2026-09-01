"""Phase-0 runner (build step 5): score the model-free arms on the frozen families.

Produces the FIRST real retrieval numbers — match / closure-by-class / contract-caller recall + closure
precision + materialized-token waste — at a matched token budget, against the pinned agentic-os tree.

Scope note (honest): this scores RETRIEVAL only. `resolution` / `resolution_efficiency` need an agent to
apply the change and a test to pass (build step 6 / P4); they are reported as null here, not faked. The
`existing_cr` and `existing_cr + oracle-hints` arms need the live Context Runtime stack and are recorded
PENDING — the whole point of the decision gate is to compare CR against these baselines, so their numbers
are the next real milestone, not something to approximate.

    python run.py --repo /mnt/backup/projects/agentic-os --budget 4000
"""
from __future__ import annotations

import argparse
import json
import os

import yaml

from resolver import build_call_graph, index_repo
from retrievers import (existing_cr_arm, existing_cr_hybrid_arm, graph_closure_arm,
                        graph_plus_hybrid_arm, naive_semantic_arm, oracle_closure_arm)
from scoring import DEP_CLASSES, Oracle, RetrievedContext, RetrievedUnit, score

HERE = os.path.dirname(os.path.abspath(__file__))
CR_ARMS = ["existing_cr", "existing_cr_oracle_hints", "existing_cr_hybrid", "existing_cr_hybrid_oracle_hints"]
# graph/union arms are SEEDED on the symbol to change (structural), not on the NL query — see graph_closure_arm.
GRAPH_ARMS = ["graph_closure", "graph_plus_hybrid"]


def _hint_text(o_ret) -> str:
    """seed symbols + required dependency CLASS names (never files) — the oracle-hints probe."""
    classes = [c for c in DEP_CLASSES if o_ret.required_by_class.get(c)]
    return f"symbols {sorted(o_ret.seed_symbols)}; required dependency classes {classes}"


def _full(index) -> RetrievedContext:
    return RetrievedContext([RetrievedUnit(s.id, s.tokens, s.file, (s.start, s.end)) for s in index.values()])


def _fmt(v):
    return "  N/A" if v is None else f"{v:5.2f}"


def run(repo: str, budget: int) -> dict:
    index = index_repo(repo)
    graph = build_call_graph(index)
    repo_total = sum(s.tokens for s in index.values())
    task_dir = os.path.join(HERE, "oracle", "tasks")
    results = {"repo": repo, "base_commit": "f8ce79c", "budget_tokens": budget,
               "repo_total_tokens": repo_total, "tasks": {}}

    for fn in sorted(os.listdir(task_dir)):
        if not fn.endswith(".yaml"):
            continue
        path = os.path.join(task_dir, fn)
        raw = yaml.safe_load(open(path))
        oracle = Oracle.load(path)
        o_ret = oracle.for_retrieval()          # score retrieval against what actually exists to be found
        query = raw.get("description", "") or ""

        arms = {
            "full": _full(index),
            "naive_semantic": naive_semantic_arm(query, o_ret.seed_symbols, index, budget),
            "oracle_closure": oracle_closure_arm(o_ret, index),
        }
        try:
            arms["existing_cr"] = existing_cr_arm(query, index, budget)
            arms["existing_cr_oracle_hints"] = existing_cr_arm(query, index, budget, hint_text=_hint_text(o_ret))
        except Exception:  # contextos not importable -> record, don't fake
            for n in ("existing_cr", "existing_cr_oracle_hints"):
                arms.pop(n, None)
        try:
            arms["existing_cr_hybrid"] = existing_cr_hybrid_arm(query, index, budget)
            arms["existing_cr_hybrid_oracle_hints"] = existing_cr_hybrid_arm(
                query, index, budget, hint_text=_hint_text(o_ret))
        except Exception:  # redevops_rag / embedder unavailable -> record, don't fake
            for n in ("existing_cr_hybrid", "existing_cr_hybrid_oracle_hints"):
                arms.pop(n, None)
        # structural arms (seeded on the symbol to change): call-graph traversal, and the graph∪hybrid union.
        arms["graph_closure"] = graph_closure_arm(o_ret.seed_symbols, index, graph, budget)
        try:
            arms["graph_plus_hybrid"] = graph_plus_hybrid_arm(o_ret.seed_symbols, query, index, graph, budget)
        except Exception:
            arms.pop("graph_plus_hybrid", None)
        scored = {name: score(o_ret, ctx, repo_total) for name, ctx in arms.items()}
        for name in CR_ARMS:
            if name not in scored:
                scored[name] = {"status": "PENDING", "reason": "Context Runtime not importable in this env"}
        results["tasks"][oracle.task_id] = {
            "declaration_hash": oracle.declaration_hash,
            "family": raw.get("family"),
            "arms": scored,
        }

        print(f"\n=== {oracle.task_id}  [{raw.get('family')}]  {oracle.declaration_hash}")
        print(f"    seed={sorted(o_ret.seed_symbols)}  contract_callers={sorted(o_ret.contract_callers)}")
        hdr = f"    {'arm':30} {'match':>6} {'closure':>8} {'contract':>9} {'prec':>6} {'mat_tok':>8} {'waste':>7}"
        print(hdr)
        order = ["full", "naive_semantic", "existing_cr", "existing_cr_oracle_hints",
                 "existing_cr_hybrid", "existing_cr_hybrid_oracle_hints",
                 "graph_closure", "graph_plus_hybrid", "oracle_closure"]
        for name in order:
            s = scored.get(name)
            if s is None:
                continue
            if s.get("status") == "PENDING":
                print(f"    {name:30} PENDING — {s['reason']}")
                continue
            print(f"    {name:30} {_fmt(s['match_recall'])} {_fmt(s['closure_recall_total'])} "
                  f"{_fmt(s['contract_caller_recall'])} {_fmt(s['closure_precision'])} "
                  f"{s['materialized_tokens']:8d} {s['materialized_token_waste']:7d}")
        # per-class closure recall for the retrieval arms (the actionable breakdown)
        for name in ["naive_semantic", "existing_cr_hybrid", "graph_closure", "graph_plus_hybrid", "oracle_closure"]:
            if name not in scored or scored[name].get("status") == "PENDING":
                continue
            by = scored[name]["closure_recall_by_class"]
            cells = "  ".join(f"{c}={_fmt(by[c]).strip()}" for c in DEP_CLASSES)
            print(f"      closure_by_class[{name}]: {cells}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/mnt/backup/projects/agentic-os")
    ap.add_argument("--budget", type=int, default=4000)
    ap.add_argument("--out", default=os.path.join(HERE, "results_phase0.json"))
    a = ap.parse_args()
    results = run(a.repo, a.budget)
    json.dump(results, open(a.out, "w"), indent=2, sort_keys=True)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
