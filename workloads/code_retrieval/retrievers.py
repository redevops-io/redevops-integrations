"""Retrieval arms (Phase 0, build step 5).

Two model-free arms that need no external stack, so the harness produces real numbers today:

- `oracle_closure`  — the ORACLE ceiling: return exactly the declared retrievable closure. Not a system you
  could ship; it measures "if retrieval were perfect, does the closure resolve?" (decision-gate row d).
- `naive_semantic`  — the honest baseline: rank every symbol by BM25 over its own source against a query
  built from the task description + seed symbol names, then fill the token budget greedily. This is what a
  plain embedding/keyword retriever approximates; it has NO notion of "which callers' contract breaks", so
  it is expected to score match-high / contract-caller-low (the H1 shape) on real code.

`existing_cr` (the Context Runtime under test) and `existing_cr + oracle-hints` are deliberately NOT here —
they need the live CR retrieval stack/endpoint. `run.py` records them as PENDING rather than faking a number.
Every arm returns a `RetrievedContext` of symbols with real token costs, so the scorer stays arm-agnostic.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from resolver import CallGraph, Sym
from scoring import RetrievedContext, RetrievedUnit

_CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]*|[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Split code + prose into lowercased subtokens: snake_case, camelCase and digits all break apart, so
    `compile_intent` and `MissionRuntime` become searchable by their parts."""
    out: list[str] = []
    for word in re.split(r"[^A-Za-z0-9]+", text):
        if not word:
            continue
        out.extend(m.group(0).lower() for m in _CAMEL.finditer(word))
    return out


def _unit(sym: Sym) -> RetrievedUnit:
    return RetrievedUnit(sym.id, sym.tokens, sym.file, (sym.start, sym.end))


def oracle_closure_arm(oracle_for_retrieval, index: dict[str, Sym]) -> RetrievedContext:
    want = oracle_for_retrieval.seed_symbols | oracle_for_retrieval.required_all | oracle_for_retrieval.contract_callers
    return RetrievedContext([_unit(index[s]) for s in sorted(want) if s in index])


class _BM25:
    def __init__(self, docs: dict[str, list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.ids = list(docs)
        self.tf = {i: Counter(docs[i]) for i in self.ids}
        self.len = {i: len(docs[i]) for i in self.ids}
        self.avg = (sum(self.len.values()) / len(self.ids)) if self.ids else 0.0
        df: Counter = Counter()
        for i in self.ids:
            df.update(set(docs[i]))
        n = len(self.ids)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def rank(self, query: list[str]) -> list[str]:
        q = [t for t in query if t in self.idf]
        scored = []
        for i in self.ids:
            tf, dl = self.tf[i], self.len[i]
            s = 0.0
            for t in q:
                f = tf.get(t, 0)
                if not f:
                    continue
                s += self.idf[t] * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / (self.avg or 1)))
            if s > 0:
                scored.append((s, i))
        # deterministic: score desc, then symbol id asc
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [i for _, i in scored]


def existing_cr_arm(query_text: str, index: dict[str, Sym], budget_tokens: int,
                    hint_text: str = "") -> RetrievedContext:
    """The arm under test: drive the REAL Context Runtime pipeline (plan -> intent -> retrieve -> rerank ->
    compress) over its InMemoryStore, one doc per symbol (chunk_id = symbol id). No LLM is invoked — retrieval
    is `build_context`, which is deterministic. In THIS environment the dense embed endpoints are down, so CR
    runs its lexical/hybrid BM25 path (its `capabilities` are bm25/vector/hybrid); that is the honest "CR as
    shipped, no plugin" here, and the run notes it.

    Fairness: we take CR's *ranked* hits and fill the SAME token budget using the SAME per-symbol token
    accounting as every other arm — so only the RANKING differs, not the budget or the accounting (Track-A).
    `hint_text` (empty for `existing_cr`; seed symbols + required dependency CLASS names, never files, for
    `existing_cr + oracle-hints`) is appended to the goal to probe the representation gap vs mechanism gap.
    """
    import os as _os

    from context_runtime import ContextRuntime, Constraints, Goal
    from context_runtime.runtime.config import Config

    docs = [{"chunk_id": sid, "filename": _os.path.basename(s.file), "text": s.source}
            for sid, s in index.items()]
    cfg = Config(top_k=max(80, len(docs)), final_k=max(80, len(docs)), target_tokens=budget_tokens)
    rt = ContextRuntime.default(docs, config=cfg)
    goal_text = query_text + (("\nRelevant to the change: " + hint_text) if hint_text else "")
    goal = Goal(text=goal_text, constraints=Constraints(max_tokens=budget_tokens))
    ctx = rt.build_context(rt.plan(goal), goal)

    units, spent = [], 0
    for h in ctx.hits:
        sym = index.get(h.chunk_id)
        if sym is None:
            continue
        if spent + sym.tokens > budget_tokens and units:
            break
        units.append(_unit(sym))
        spent += sym.tokens
        if spent >= budget_tokens:
            break
    return RetrievedContext(units)


_RAG_STORE_CACHE: dict = {}


def _rag_store(index: dict[str, Sym]):
    """Build (once, cached) a redevops_rag DuckDB Store over the symbol corpus with the SHIPPED default
    embedder (bge-small-en, 384-d, in-process CPU — no embed endpoint needed; the heavier Nemotron-embed at
    :8013 is an opt-in via REDEVOPS_RAG_NEMOTRON_URL). One chunk per symbol, id = symbol id."""
    key = (id(index), len(index))
    if key in _RAG_STORE_CACHE:
        return _RAG_STORE_CACHE[key]
    import os as _os

    from redevops_rag.embed import Embedder
    from redevops_rag.store import Store

    emb = Embedder()
    store = Store(emb, ":memory:")
    sids = list(index)
    vecs = emb.encode([index[s].source for s in sids])
    chunks = [{"id": s, "document_id": s, "filename": _os.path.basename(index[s].file),
               "chunk_index": 0, "text": index[s].source, "embedding": v}
              for s, v in zip(sids, vecs)]
    store.add_chunks(chunks)
    store.reindex_fts()
    _RAG_STORE_CACHE[key] = store
    return store


def existing_cr_hybrid_arm(query_text: str, index: dict[str, Sym], budget_tokens: int,
                           hint_text: str = "") -> RetrievedContext:
    """The DENSE+SPARSE arm: redevops_rag `hybrid_search` = vector (bge-small) + BM25 -> RRF fusion ->
    recency/keyword boosts. This is CR's shipped hybrid retrieval, actually using dense embeddings (unlike
    `existing_cr`, whose InMemoryStore 'hybrid' is lexical-only). Same budget-fill + token accounting as every
    other arm, so only the ranking differs (Track-A)."""
    from redevops_rag.retrieve import hybrid_search

    store = _rag_store(index)
    q = query_text + (("\nRelevant to the change: " + hint_text) if hint_text else "")
    hits = hybrid_search(store, q, limit=80, pool=120)
    units, spent = [], 0
    for h in hits:
        sid = h.get("document_id") or h.get("id")
        sym = index.get(sid)
        if sym is None:
            continue
        if spent + sym.tokens > budget_tokens and units:
            break
        units.append(_unit(sym))
        spent += sym.tokens
        if spent >= budget_tokens:
            break
    return RetrievedContext(units)


def graph_closure_arm(seed_symbols: set[str], index: dict[str, Sym], graph: CallGraph,
                      budget_tokens: int, up_hops: int = 2, down_hops: int = 1) -> RetrievedContext:
    """The graph-traversal capability (Phase-1 candidate). Given the SEED symbol(s), walk the static call
    graph: UP the caller edges `up_hops` levels (recovers direct AND transitive callers — the classes content
    retrieval misses) and DOWN the callee edges `down_hops` level (helpers the change may touch). Budget-fill
    in priority order: seed, then callers nearest-first, then callees — so the closure members that MUST change
    are kept before optional context.

    Note on regime: this arm is SEEDED on the symbol to change (structural), unlike the content arms that work
    from the NL description. That is realistic — an agent first locates the seed the task names, then needs its
    change-closure — but it is a different input, so compare it as "does traversal recover the closure classes
    content retrieval misses," not as a like-for-like ranking of the same query."""
    # BFS up the caller graph, tracking hop distance (nearest wins on ties).
    order: list[str] = [s for s in sorted(seed_symbols) if s in index]
    seen = set(order)
    frontier = set(order)
    for _ in range(up_hops):
        nxt: set[str] = set()
        for s in frontier:
            for caller in graph.callers.get(s, ()):
                if caller not in seen and caller in index:
                    nxt.add(caller)
        for c in sorted(nxt):
            seen.add(c)
            order.append(c)
        frontier = nxt
    # then callees (down), one level per seed's reachable set
    callee_frontier = set(order)
    for _ in range(down_hops):
        nxt = set()
        for s in callee_frontier:
            for callee in graph.callees.get(s, ()):
                if callee not in seen and callee in index:
                    nxt.add(callee)
        for c in sorted(nxt):
            seen.add(c)
            order.append(c)
        callee_frontier = nxt

    units, spent = [], 0
    for sid in order:
        sym = index[sid]
        if spent + sym.tokens > budget_tokens and units:
            break
        units.append(_unit(sym))
        spent += sym.tokens
        if spent >= budget_tokens:
            break
    return RetrievedContext(units)


_BIG = 10 ** 9


def graph_plus_hybrid_arm(seed_symbols: set[str], query_text: str, index: dict[str, Sym],
                          graph: CallGraph, budget_tokens: int) -> RetrievedContext:
    """The complementary UNION: structural traversal (graph) ∪ content retrieval (dense hybrid). Graph recovers
    self-method caller chains (incl. transitive) that content retrieval misses; hybrid recovers class-
    instantiation / non-self-receiver / type edges that the graph misses. We take the graph closure FIRST
    (the must-change members), then let hybrid fill the remaining budget — so neither representation's blind
    spot sinks the closure. Same budget + token accounting as every other arm."""
    graph_units = graph_closure_arm(seed_symbols, index, graph, _BIG).units
    hybrid_units = existing_cr_hybrid_arm(query_text, index, _BIG).units
    seen: set[str] = set()
    merged, spent = [], 0
    for u in [*graph_units, *hybrid_units]:
        if u.symbol in seen:
            continue
        seen.add(u.symbol)
        if spent + u.tokens > budget_tokens and merged:
            break
        merged.append(u)
        spent += u.tokens
        if spent >= budget_tokens:
            break
    return RetrievedContext(merged)


def naive_semantic_arm(query_text: str, seed_symbols: set[str], index: dict[str, Sym],
                       budget_tokens: int) -> RetrievedContext:
    docs = {sid: tokenize(sym.source) for sid, sym in index.items()}
    bm = _BM25(docs)
    query = tokenize(query_text) + [p for s in seed_symbols for p in tokenize(s.split(":")[-1])]
    ranked = bm.rank(query)
    units, spent = [], 0
    for sid in ranked:
        sym = index[sid]
        if spent + sym.tokens > budget_tokens and units:
            break
        units.append(_unit(sym))
        spent += sym.tokens
        if spent >= budget_tokens:
            break
    return RetrievedContext(units)
