"""Clinical domain — drug–drug interaction (DDI) closure (real, deterministic).

Prescribing a drug means considering every other drug it interacts with. That interaction set is a deterministic
closure over a real pharmacological graph (DDIMDL, derived from DrugBank 5.1.3). Units = drugs; seed = a drug;
gold = the drugs it interacts with (within a bounded universe of the most-connected drugs + distractors);
structural edges = the interaction graph.

IMPORTANT CAVEAT: unlike the other domains, DDI drugs carry no descriptive *text* — only a name and structured
target/enzyme codes. So the content (bge-over-name) leg is near-random *by construction*: this domain is the
**pure-structural extreme** of the axis (the counterpart to legal/scientific's content-led extreme), included to
show the same composition handles a domain whose closure is entirely relational. It is not a text-retrieval test.

Data: DDIMDL event.db (YifanDengWHU/DDIMDL). Cached to data/ddi/ on first run.
"""
from __future__ import annotations

import os
import sqlite3
import urllib.request

from common import ClosureCase, Unit, approx_tokens

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "data", "ddi", "event.db")
URL = "https://github.com/YifanDengWHU/DDIMDL/raw/master/event.db"


def _load_graph():
    if not os.path.exists(DB):
        os.makedirs(os.path.dirname(DB), exist_ok=True)
        data = urllib.request.urlopen(urllib.request.Request(URL, headers={"User-Agent": "M"}), timeout=150).read()
        open(DB, "wb").write(data)
    con = sqlite3.connect(DB)
    edges: dict[str, set[str]] = {}
    for a, b in con.execute("SELECT name1, name2 FROM event"):
        edges.setdefault(a, set()).add(b)
        edges.setdefault(b, set()).add(a)
    con.close()
    return edges


def cases(rank_lo: int = 150, rank_hi: int = 400, min_gold: int = 5, max_gold: int = 25) -> list[ClosureCase]:
    edges = _load_graph()
    # the DDI graph is very dense (median degree ~120/570), so the most-connected drugs have huge closures.
    # Use a MID-connectivity band → bounded, sparse-enough closures with real distractors in the universe.
    universe = sorted(edges, key=lambda d: -len(edges[d]))[rank_lo:rank_hi]
    uset = set(universe)
    units = [Unit(id=d, text=d, tokens=approx_tokens(d) + 3) for d in universe]   # name-only text (see caveat)
    adjacency = {d: (edges[d] & uset) for d in universe}
    out = []
    for d in universe:
        gold = edges[d] & uset
        gold.discard(d)
        if not (min_gold <= len(gold) <= max_gold):     # skip hubs (closure too large) and leaves
            continue
        out.append(ClosureCase(case_id=f"ddi-{d}", query=d, seed=[d],
                               gold=set(gold), units=units, adjacency=adjacency))
    return out


if __name__ == "__main__":
    cs = cases()
    print(f"clinical-interaction (DDI): {len(cs)} drug cases (bounded universe)")
    if cs:
        ex = cs[0]
        print(f"  e.g. {ex.case_id}: gold {len(ex.gold)} interacting drugs, {len(ex.units)} drugs in universe")
