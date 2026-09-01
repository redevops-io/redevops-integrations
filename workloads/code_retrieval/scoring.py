"""Closure instrumentation + scorer (Phase 0, build step 3).

Given a task's FROZEN oracle declaration and what an arm actually retrieved, compute every metric named in
`frozen_spec.yaml` — per-class recall, total closure recall, contract-caller recall, closure precision,
materialized-token waste, context efficiency. Resolution-efficiency needs the run-time resolution outcome
and the materialized-token denominator this scorer emits.

The scorer is deliberately arm-agnostic: an arm reports the SYMBOLS it surfaced (each with its file, line
range and token cost). Mapping retrieved chunks -> symbols is the arm's / a shared resolver's job, not the
scorer's — so the scorer stays a pure, testable function of (oracle, retrieved).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import yaml

# the seven dependency classes, frozen in frozen_spec.yaml (order fixed for stable reporting)
DEP_CLASSES = ["seed_definition", "direct_caller", "transitive_caller", "type_schema",
               "import_export", "test", "config_build"]


@dataclass(frozen=True)
class RetrievedUnit:
    symbol: str            # canonical "module:symbol" id the arm surfaced
    tokens: int            # materialized token cost of this unit
    file: str = ""         # anchor (for P4); not used by recall/precision
    line_range: tuple[int, int] | None = None


@dataclass
class RetrievedContext:
    units: list[RetrievedUnit] = field(default_factory=list)

    @property
    def symbols(self) -> set[str]:
        return {u.symbol for u in self.units}

    @property
    def materialized_tokens(self) -> int:
        return sum(u.tokens for u in self.units)


@dataclass
class Oracle:
    task_id: str
    seed_symbols: set[str]
    required_by_class: dict[str, set[str]]     # class -> required symbols (empty set = explicitly NOT required)
    contract_callers: set[str]
    declaration_hash: str
    creates: set[str] = field(default_factory=set)  # symbols the patch CREATES (new tests etc.) — not retrievable

    @property
    def required_all(self) -> set[str]:
        out: set[str] = set()
        for s in self.required_by_class.values():
            out |= s
        return out

    def for_retrieval(self) -> "Oracle":
        """A retrieval arm can only surface symbols that ALREADY exist. Created symbols (a new test) are a
        completeness/application concern, not a retrieval one — remove them so they don't count against
        retrieval recall (that would penalize retrieval for not finding code the patch invents)."""
        c = self.creates
        return Oracle(
            task_id=self.task_id,
            seed_symbols=self.seed_symbols - c,
            required_by_class={k: v - c for k, v in self.required_by_class.items()},
            contract_callers=self.contract_callers - c,
            declaration_hash=self.declaration_hash,
            creates=set(),
        )

    @staticmethod
    def load(path: str) -> "Oracle":
        with open(path) as f:
            d = yaml.safe_load(f)
        rc = d.get("required_closure", {}) or {}
        by_class = {c: set(rc.get(c, []) or []) for c in DEP_CLASSES}
        return Oracle(
            task_id=d["task_id"],
            seed_symbols=set(d.get("seed_symbols", []) or []),
            required_by_class=by_class,
            contract_callers=set(d.get("contract_callers", []) or []),
            declaration_hash=d.get("declaration_hash", ""),
            creates=set(d.get("creates", []) or []),
        )


def _recall(retrieved: set[str], required: set[str]) -> float | None:
    # An empty required set is an EXPLICIT claim the class isn't required -> recall is N/A (not a free 1.0).
    if not required:
        return None
    return len(retrieved & required) / len(required)


def score(oracle: Oracle, ctx: RetrievedContext, repo_total_tokens: int | None = None) -> dict:
    got = ctx.symbols
    req = oracle.required_all
    by_class = {c: _recall(got, oracle.required_by_class[c]) for c in DEP_CLASSES}

    # precision / waste: of what the arm surfaced, how much is actually required?
    hit = got & req
    precision = (len(hit) / len(got)) if got else None
    waste_tokens = sum(u.tokens for u in ctx.units if u.symbol not in req)

    return {
        "task_id": oracle.task_id,
        "declaration_hash": oracle.declaration_hash,
        "match_recall": _recall(got, oracle.seed_symbols),
        "closure_recall_total": _recall(got, req),
        "closure_recall_by_class": by_class,
        "contract_caller_recall": _recall(got, oracle.contract_callers),
        "closure_precision": precision,
        "materialized_tokens": ctx.materialized_tokens,
        "materialized_token_waste": waste_tokens,
        "context_efficiency": (ctx.materialized_tokens / repo_total_tokens) if repo_total_tokens else None,
        # resolution + resolution_efficiency are filled in at run time (need the pass/fail outcome):
        # resolution_efficiency = resolved(0/1) / (materialized_tokens/1000), reported subject to required-edge recall.
    }
