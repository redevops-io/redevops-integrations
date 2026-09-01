"""Common result classification for every framework/cloud integration (both plans, Section 4).

This exists so an integration benchmark is a CONFORMANCE exercise, not a marketing scorecard. Every measured
comparison of `framework-native` vs `framework + ReDevOps` is classified into one of five buckets.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass


class ResultClass(str, enum.Enum):
    PARITY = "PARITY"                                   # framework + ReDevOps preserves the framework outcome
    FRAMEWORK_NATIVE_ADVANTAGE = "FRAMEWORK_NATIVE_ADVANTAGE"   # framework already solves it better natively
    REDEVOPS_DELTA = "REDEVOPS_DELTA"                   # ReDevOps adds a measurable guarantee / optimization
    EXPECTED_IMPLEMENTATION_DIFFERENCE = "EXPECTED_IMPLEMENTATION_DIFFERENCE"  # syntax/latency differ, semantics hold
    BUG = "BUG"                                         # unexplained semantic divergence


@dataclass
class Finding:
    experiment: str
    metric: str
    native: float | str | None
    with_redevops: float | str | None
    classification: ResultClass
    note: str = ""


def classify(native, with_redevops, *, higher_is_better=True, guarantee_added=False,
             semantics_preserved=True, tol=1e-9) -> ResultClass:
    """Deterministic classifier. `guarantee_added` = ReDevOps enforced something the native path cannot
    express (authority, replay, abstention) — that is a REDEVOPS_DELTA even at equal task score."""
    if not semantics_preserved:
        return ResultClass.BUG
    if guarantee_added:
        return ResultClass.REDEVOPS_DELTA
    if native is None or with_redevops is None or isinstance(native, str):
        return ResultClass.EXPECTED_IMPLEMENTATION_DIFFERENCE
    d = with_redevops - native
    if abs(d) <= tol:
        return ResultClass.PARITY
    better = d > 0 if higher_is_better else d < 0
    return ResultClass.REDEVOPS_DELTA if better else ResultClass.FRAMEWORK_NATIVE_ADVANTAGE
