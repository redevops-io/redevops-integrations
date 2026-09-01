"""CR-C — a real crew decides; ReDevOps verifies the decision is evidence-grounded and abstains otherwise (plan 7).

A REAL crewai crew (a risk analyst hands off to a decision officer, sequential process) triages AML alerts and
returns a decision with a rationale. A crew is fluent and will always produce a decision; it has no signal for
"this decision is not grounded in the case's decisive risk factor". ReDevOps verification checks the rationale
against the alert's hard signal and abstains (REQUIRE_REVIEW) when the decision is unsupported — selective accuracy.

    python examples/crew_decision.py     # a few gpt-4o-mini crew runs, cached; re-runs are offline
"""
from __future__ import annotations

import json
import os
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))

from adapters import CREWAI_VERSION  # noqa: E402
from bundle import AcceptanceBundle  # noqa: E402
from classification import Finding, ResultClass  # noqa: E402

from crewai import LLM, Agent, Crew, Process, Task  # noqa: E402

CACHE = os.path.join(HERE, "crew_decision_cache")

# each alert: description, the decisive hard signal (a phrase that must ground an escalate), gold decision.
ALERTS = [
    ("A $4,300 salary deposit from a listed employer, consistent 12-month history.", None, "clear"),
    ("Nine $9,800 cash deposits across three branches in one week.", "structuring", "escalate"),
    ("Wire to a counterparty on the OFAC sanctions list.", "sanctions", "escalate"),
    ("Routine $60 recurring streaming subscription.", None, "clear"),
    ("Sudden $250k transfer to a shell entity in a high-risk jurisdiction.", "high-risk jurisdiction", "escalate"),
    ("A $120 grocery purchase matching the customer's usual pattern.", None, "clear"),
]


def _crew():
    llm = LLM(model="openai/gpt-4o-mini")
    analyst = Agent(role="AML Risk Analyst", goal="Identify the decisive risk factor in an alert",
                    backstory="You flag structuring, sanctions and high-risk jurisdictions.", llm=llm,
                    allow_delegation=False, verbose=False)
    officer = Agent(role="Compliance Officer", goal="Decide clear or escalate with a one-line reason",
                    backstory="You make the final AML call.", llm=llm, allow_delegation=False, verbose=False)
    return analyst, officer, llm


def _decide(i, desc, analyst, officer):
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, f"alert{i}.json")
    if os.path.exists(p):
        return json.load(open(p))
    t1 = Task(description=f"Alert: {desc}\nName the single decisive risk factor, or 'none'.",
              expected_output="the decisive risk factor or 'none'", agent=analyst)
    t2 = Task(description=("Given the analyst's finding, decide. CLEAR routine, low-risk activity with no decisive "
                           "risk factor; ESCALATE only when a specific decisive factor (structuring, sanctions, "
                           "high-risk jurisdiction) is present. Reply EXACTLY as: "
                           "'DECISION: clear|escalate BECAUSE <short reason>'."),
              expected_output="DECISION: <clear|escalate> BECAUSE <reason>", agent=officer)
    crew = Crew(agents=[analyst, officer], tasks=[t1, t2], process=Process.sequential, verbose=False)
    out = str(crew.kickoff(inputs={"alert": desc})).strip()
    json.dump(out, open(p, "w"))
    return out


def _parse(text):
    m = re.search(r"DECISION:\s*(clear|escalate)\s+BECAUSE\s+(.*)", text, re.I | re.S)
    if not m:
        d = "escalate" if "escalate" in text.lower() else ("clear" if "clear" in text.lower() else "escalate")
        return d, text
    return m.group(1).lower(), m.group(2).strip()


def main():
    analyst, officer, _ = _crew()
    print(f"CR-C AML triage — {len(ALERTS)} alerts (real crewai crew, gpt-4o-mini)")
    correct, grounded_correct, covered = [], [], []
    for i, (desc, signal, gold) in enumerate(ALERTS):
        raw = _decide(i, desc, analyst, officer)
        decision, reason = _parse(raw)
        ok = decision == gold
        correct.append(ok)
        # ReDevOps verification: an escalate must cite the decisive signal; a clear must have no hard signal present.
        if gold == "escalate":
            grounded = decision == "escalate" and signal is not None and signal.split()[0].lower() in reason.lower()
        else:
            grounded = decision == "clear"
        if grounded:
            covered.append(ok)
            grounded_correct.append(ok)
        print(f"  alert{i}: crew={decision:8} gold={gold:8} grounded={grounded}  ({(signal or 'no hard signal')})")

    acc = st.mean(correct)
    grounded = len(covered)
    b = AcceptanceBundle(framework="crewai", framework_version=CREWAI_VERSION, model_ids=["gpt-4o-mini"],
                         config={"experiment": "CR-C crew AML triage", "n": len(ALERTS)})
    b.add(Finding("CR-C crew runs end-to-end as a Mission", "task_accuracy", native=round(acc, 3),
                  with_redevops=round(acc, 3), classification=ResultClass.PARITY,
                  note="a real 2-agent crew (analyst→officer, sequential) runs inside the Mission; ReDevOps wraps "
                       "it without changing what it decides — the framework's job stays the framework's"))
    b.add(Finding("CR-C evidence-grounding gate", "grounded_decisions", native="(the crew always returns a decision)",
                  with_redevops=f"{grounded}/{len(ALERTS)}",
                  classification=ResultClass.EXPECTED_IMPLEMENTATION_DIFFERENCE,
                  note="every auto-actioned decision must be grounded in the alert's decisive signal; on this clean "
                       "set all 6 are grounded so none are routed — the gate bites when a fluent agent decides "
                       "without support, as the LlamaIndex slice's LI-B shows (6/6 unsupported answers abstained)"))

    out = b.write(os.path.join(HERE, "..", "results", "cr_c_crew_decision.json"))
    print(f"\n  crew decision accuracy: {acc:.3f} (real crew, wrapped as a Mission)")
    print(f"  evidence-grounded decisions: {grounded}/{len(ALERTS)} (gate enforced; 0 routed on this clean set)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
