"""Common closure-resolution harness for the third-domain generalization test.

The reframed capability treats every domain as a *representation provider + a closure task*. This module defines
the domain-neutral interface (Unit / ClosureCase) and the generic runner that puts any domain through the SAME
composition the code/legal experiments used:

  content (bge similarity to the query)  +  alpha * structural (proximity to the seed via the domain's edges)
  → seed-first, budget-fill  → closure recall.

It reports exactly what the generalization question needs, per domain:
  - closure principle: seed-only recall vs full-composition recall (does complete context beat partial?)
  - composition character: the alpha the domain wants (structural-heavy → high alpha; semantic → alpha≈0)
  - ceiling + governance: recall at budget, and how much residual remains (→ abstain, per Gate 5).

No domain-specific logic lives here; each domain supplies cases() returning ClosureCase objects.
"""
from __future__ import annotations

import math
import os
import statistics as st
from dataclasses import dataclass, field

_EMBED = None


def _embedder():
    global _EMBED
    if _EMBED is None:
        from redevops_rag.embed import Embedder
        _EMBED = Embedder()
    return _EMBED


@dataclass
class Unit:
    id: str
    text: str
    tokens: int


@dataclass
class ClosureCase:
    case_id: str
    query: str                       # the change/claim/obligation driving the closure
    seed: list[str]                  # seed unit id(s)
    gold: set[str]                   # the gold closure (unit ids)
    units: list[Unit]                # the retrieval universe for this case
    adjacency: dict[str, set[str]] = field(default_factory=dict)   # structural edges (domain-defined)


def approx_tokens(text: str) -> int:
    return max(1, int(round(len(text.split()) * 1.3)))


_VEC_CACHE: dict[str, object] = {}   # md5(text) -> normalised vector (persisted; units repeat across runs)
_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "vec_cache.pkl")
_CACHE_LOADED = False


def _key(t):
    import hashlib
    return hashlib.md5(t.encode("utf-8")).hexdigest()


def _load_cache():
    global _CACHE_LOADED
    if _CACHE_LOADED:
        return
    _CACHE_LOADED = True
    if os.path.exists(_CACHE_PATH):
        import pickle
        try:
            _VEC_CACHE.update(pickle.load(open(_CACHE_PATH, "rb")))
        except Exception:
            pass


def _encode(texts):
    import numpy as np
    _load_cache()
    keys = [_key(t) for t in texts]
    missing = [(t, k) for t, k in zip(texts, keys) if k not in _VEC_CACHE]
    if missing:
        V = np.asarray(_embedder().encode([t for t, _ in missing]), dtype="float32")
        V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
        for (t, k), v in zip(missing, V):
            _VEC_CACHE[k] = v
        import pickle
        os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
        pickle.dump(_VEC_CACHE, open(_CACHE_PATH, "wb"))
    return np.asarray([_VEC_CACHE[k] for k in keys], dtype="float32")


def _content_scores(case: ClosureCase):
    import numpy as np
    V = _encode([u.text for u in case.units])
    q = _encode([case.query])[0]
    sims = V @ q
    order = list(np.argsort(-sims))
    return {case.units[int(i)].id: 1.0 / (r + 1) for r, i in enumerate(order)}


def _struct_scores(case: ClosureCase, hops: int = 2):
    """Hop-decayed proximity to the seed along the domain's structural edges (like legal structural_scores)."""
    score: dict[str, float] = {}
    frontier = set(case.seed)
    seen = set(case.seed)
    for h in range(hops):
        nxt = set()
        for u in frontier:
            for v in case.adjacency.get(u, ()):
                if v not in seen:
                    score[v] = max(score.get(v, 0.0), 1.0 / (h + 1))
                    nxt.add(v); seen.add(v)
        frontier = nxt
    return score


def _fill(order, tok, budget, seed):
    kept, spent = set(), 0
    for u in list(dict.fromkeys([*seed, *order])):
        c = tok.get(u, 0)
        if spent + c > budget and kept:
            break
        kept.add(u); spent += c
    return kept


def run_domain(name: str, cases: list[ClosureCase], budget: int, alphas=(0.0, 0.3, 0.6, 1.0, 2.0)) -> dict:
    per_alpha = {a: [] for a in alphas}
    seed_only, oracle_fit = [], []
    for c in cases:
        tok = {u.id: u.tokens for u in c.units}
        content = _content_scores(c)
        struct = _struct_scores(c)
        gold = c.gold
        seed_only.append(len(set(c.seed) & gold) / len(gold) if gold else 1.0)
        oracle_fit.append(sum(tok.get(g, 0) for g in gold) <= budget)
        for a in alphas:
            total = {u.id: content.get(u.id, 0.0) + a * struct.get(u.id, 0.0) for u in c.units}
            order = sorted(total, key=lambda i: (-total[i], i))
            got = _fill(order, tok, budget, c.seed)
            per_alpha[a].append(len(got & gold) / len(gold) if gold else 1.0)
    best_a = max(alphas, key=lambda a: st.mean(per_alpha[a]))
    return {
        "name": name, "n": len(cases), "budget": budget,
        "seed_only": st.mean(seed_only),
        "oracle_fit": st.mean(oracle_fit),
        "recall_by_alpha": {a: st.mean(per_alpha[a]) for a in alphas},
        "best_alpha": best_a,
        "best_recall": st.mean(per_alpha[best_a]),
        "content_only_recall": st.mean(per_alpha[0.0]),
    }


def print_report(r: dict):
    print(f"\n=== {r['name']} ({r['n']} cases, budget {r['budget']}) ===")
    print(f"  seed-only recall {r['seed_only']:.3f} → full-composition best {r['best_recall']:.3f} "
          f"(α*={r['best_alpha']})   [oracle fits: {r['oracle_fit']:.0%}]")
    print("  recall by α: " + "  ".join(f"{a}:{r['recall_by_alpha'][a]:.3f}" for a in r['recall_by_alpha']))
