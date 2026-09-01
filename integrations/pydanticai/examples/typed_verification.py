"""PA-D — Pydantic schema validity is NOT ReDevOps semantic verification (frameworks plan 7 / "5.4").

PydanticAI's typed boundary guarantees a well-typed output. Structure alone does not establish that the output is
*supported by evidence*, *fresh*, *authorized*, or *policy-admissible* — those are runtime verification/governance,
not properties of the type. We build outputs that are perfectly schema-valid yet must not be acted on, and show the
two layers are orthogonal: Pydantic says "valid"; ReDevOps says "not admissible", with a reason. This is the
flagship differentiation for PydanticAI.
Deterministic — no LLM — so it isolates the mechanism.

    python examples/typed_verification.py
"""
from __future__ import annotations

import os
import sys
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))

from adapters import PA_VERSION  # noqa: E402
from bundle import AcceptanceBundle  # noqa: E402
from classification import Finding, ResultClass  # noqa: E402

from pydantic import BaseModel, Field, ValidationError  # noqa: E402


class StrategyRecommendation(BaseModel):
    """The application's local typed output — PydanticAI validates this shape."""
    security: str
    action: str = Field(pattern="^(buy|sell|hold)$")
    rationale: str
    evidence_claim: str
    evidence_date: date
    required_permission: str


# ---- the ReDevOps runtime verification / governance layer (orthogonal to schema validity) ----
EVIDENCE = "Rising rates pressure long-duration bonds; the fund holds 80% 20y+ Treasuries."
GRANTS = {"portfolio:rebalance"}
RESTRICTED = {"ACME"}                                    # e.g. insider/restricted list
AS_OF = date(2026, 8, 31)
MAX_AGE_DAYS = 90


def redevops_admissible(rec: StrategyRecommendation) -> tuple[bool, str]:
    if rec.evidence_claim.lower() not in EVIDENCE.lower():
        return False, "verification: recommendation not supported by retrieved evidence"
    if (AS_OF - rec.evidence_date).days > MAX_AGE_DAYS:
        return False, "verification: cited evidence is stale (beyond freshness bound)"
    if rec.required_permission not in GRANTS:
        return False, "authority: mission has no grant for the proposed action"
    if rec.security in RESTRICTED:
        return False, "governance: security is on the restricted list — deny-by-default"
    return True, "admissible"


def main():
    # every case below is SCHEMA-VALID (Pydantic accepts it) — but only one is admissible.
    cases = [
        ("unsupported", StrategyRecommendation(security="TLT", action="sell",
            rationale="hunch", evidence_claim="crypto will rally", evidence_date=date(2026, 8, 20),
            required_permission="portfolio:rebalance")),
        ("stale-evidence", StrategyRecommendation(security="TLT", action="sell",
            rationale="rates", evidence_claim="Rising rates pressure long-duration bonds",
            evidence_date=date(2026, 1, 1), required_permission="portfolio:rebalance")),
        ("missing-authority", StrategyRecommendation(security="TLT", action="sell",
            rationale="rates", evidence_claim="Rising rates pressure long-duration bonds",
            evidence_date=date(2026, 8, 25), required_permission="portfolio:liquidate")),
        ("governance-denied", StrategyRecommendation(security="ACME", action="buy",
            rationale="tip", evidence_claim="Rising rates pressure long-duration bonds",
            evidence_date=date(2026, 8, 25), required_permission="portfolio:rebalance")),
        ("admissible", StrategyRecommendation(security="TLT", action="sell",
            rationale="rates", evidence_claim="Rising rates pressure long-duration bonds",
            evidence_date=date(2026, 8, 25), required_permission="portfolio:rebalance")),
    ]

    b = AcceptanceBundle(framework="pydantic-ai", framework_version=PA_VERSION,
                         config={"experiment": "PA-D typed-validity-vs-verification"})
    print("PA-D — Pydantic validity vs ReDevOps admissibility:")
    print(f"  {'case':18} {'pydantic_valid':>14} {'redevops_admissible':>20}  reason")
    for name, rec in cases:
        try:
            StrategyRecommendation.model_validate(rec.model_dump())   # schema validity
            pyd_valid = True
        except ValidationError:
            pyd_valid = False
        admissible, reason = redevops_admissible(rec)
        print(f"  {name:18} {str(pyd_valid):>14} {str(admissible):>20}  {reason}")
        if name == "admissible":
            cls = ResultClass.PARITY if (pyd_valid and admissible) else ResultClass.BUG
            note = "both layers agree the output may proceed"
        else:
            # schema-valid but rejected by the runtime = the guarantee Pydantic cannot provide
            cls = ResultClass.REDEVOPS_DELTA if (pyd_valid and not admissible) else ResultClass.BUG
            note = reason
        b.add(Finding(f"PA-D {name}", "valid_but_admissible", native=f"pydantic_valid={pyd_valid}",
                      with_redevops=f"admissible={admissible}", classification=cls, note=note))

    out = b.write(os.path.join(HERE, "..", "results", "pa_d_typed_verification.json"))
    print(f"\n  {sum(1 for f in b.findings if f.classification == ResultClass.REDEVOPS_DELTA)} REDEVOPS_DELTA "
          f"(schema-valid outputs the runtime correctly refuses to act on)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
