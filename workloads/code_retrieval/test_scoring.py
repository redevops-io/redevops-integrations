"""Validate the closure scorer against the frozen example oracle with synthetic arms.

No real retriever is wired yet (that is build step 5). These synthetic arms prove the INSTRUMENTATION
reports the metrics `frozen_spec.yaml` names — in particular the H1 shape a naive arm must show: high MATCH
recall but low CLOSURE and CONTRACT-CALLER recall.
"""
import os

from scoring import Oracle, RetrievedContext, RetrievedUnit, score

ORACLE = os.path.join(os.path.dirname(__file__), "oracle", "tasks", "cap_metadata_propagation.yaml")

CAP = "agentic_os.mission.types:CapabilitySpec"
NODE = "agentic_os.mission.types:Node"
COMPILE = "agentic_os.mission.compiler:compile_intent"
BUILDER = "agentic_os.mission.operator_sdk:capability"
EXEC = "agentic_os.mission.runtime:MissionRuntime._execute"
TEST = "tests:test_retry_budget_threads_and_is_honoured"


def _units(symbols, tok=100):
    return RetrievedContext([RetrievedUnit(s, tok) for s in symbols])


def test_oracle_loads_frozen_declaration():
    o = Oracle.load(ORACLE)
    assert o.declaration_hash.startswith("sha256:")
    assert o.required_all == {CAP, NODE, COMPILE, BUILDER, EXEC, TEST}
    assert o.contract_callers == {COMPILE, BUILDER}
    # empty classes are explicit "not required" -> recall N/A, not a free 1.0
    assert o.required_by_class["import_export"] == set()


def test_oracle_closure_arm_is_perfect():
    o = Oracle.load(ORACLE)
    r = score(o, _units(o.required_all), repo_total_tokens=100_000)
    assert r["match_recall"] == 1.0
    assert r["closure_recall_total"] == 1.0
    assert r["contract_caller_recall"] == 1.0
    assert r["closure_precision"] == 1.0
    assert r["materialized_token_waste"] == 0
    # empty required classes report None (N/A), not 1.0
    assert r["closure_recall_by_class"]["import_export"] is None
    assert r["closure_recall_by_class"]["direct_caller"] == 1.0


def test_naive_arm_shows_the_H1_shape():
    """Naive semantic finds the seed + a callee it stumbled on + noise, but MISSES the contract callers and
    the covering test. The scorer must show high match recall yet low closure / contract-caller recall."""
    o = Oracle.load(ORACLE)
    retrieved = _units([CAP, NODE, EXEC, "some.pkg:noise_a", "some.pkg:noise_b"])
    r = score(o, retrieved, repo_total_tokens=100_000)
    assert r["match_recall"] == 1.0                                   # both seed symbols found
    assert r["contract_caller_recall"] == 0.0                         # missed compile + builder
    assert abs(r["closure_recall_total"] - 3 / 6) < 1e-9              # 3 of 6 required
    assert abs(r["closure_precision"] - 3 / 5) < 1e-9                 # 3 required of 5 retrieved
    assert r["materialized_token_waste"] == 200                       # the 2 noise units
    assert r["closure_recall_by_class"]["direct_caller"] == 0.0       # names the missed edge class
    assert r["closure_recall_by_class"]["transitive_caller"] == 1.0


def test_over_retrieval_is_penalized_by_precision_not_recall():
    o = Oracle.load(ORACLE)
    noise = [f"pkg:n{i}" for i in range(10)]
    r = score(o, _units(list(o.required_all) + noise), repo_total_tokens=100_000)
    assert r["closure_recall_total"] == 1.0                           # recall alone can't see the waste
    assert abs(r["closure_precision"] - 6 / 16) < 1e-9               # but precision does
    assert r["materialized_token_waste"] == 1000                      # 10 noise units
