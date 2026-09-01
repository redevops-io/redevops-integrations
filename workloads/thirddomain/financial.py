"""Financial/legal-regulatory domain — GDPR cross-reference closure (real, deterministic).

A regulation article does not stand alone: editing it means considering every article it explicitly cross-
references ("in accordance with Article 6", "referred to in Article 9(2)"). That cross-reference set is a
deterministic closure — the regulatory analogue of code's caller closure, and a *structural* domain (the closure
is defined by explicit links, not by meaning). Units = the 99 GDPR articles; seed = an article with >=3
cross-references; gold = the articles it references; structural edges = the cross-reference graph. Content
retrieval (article-text similarity) is a weak signal here — cross-referenced articles are often not textually
similar — so recovering the closure needs the structural leg, which is exactly the point.

Data: GDPRtEXT structured GDPR JSON (coolharsh55/GDPRtEXT). Cached to data/gdpr/ on first run.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request

from common import ClosureCase, Unit, approx_tokens

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data", "gdpr", "gdpr.json")
URL = "https://raw.githubusercontent.com/coolharsh55/GDPRtEXT/master/gdpr.json"
_XREF = re.compile(r"Article\s+(\d+)")


def _load():
    if not os.path.exists(CACHE):
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        data = urllib.request.urlopen(urllib.request.Request(URL, headers={"User-Agent": "M"}), timeout=90).read()
        open(CACHE, "wb").write(data)
    return json.load(open(CACHE))


def _article_text(node) -> str:
    parts = [node.get("title", "")]

    def walk(n):
        if isinstance(n, dict):
            if n.get("text"):
                parts.append(n["text"])
            for k in ("contents", "points", "subpoints"):
                for ch in n.get(k, []) or []:
                    walk(ch)
    walk(node)
    return " ".join(p for p in parts if p)


def _articles(j):
    arts = {}
    for ch in j["chapters"]:
        stack = list(ch.get("contents", []) or [])
        while stack:
            n = stack.pop()
            if isinstance(n, dict):
                if n.get("type") == "article" and n.get("number"):
                    arts[str(n["number"])] = _article_text(n)
                stack.extend(n.get("contents", []) or [])
                stack.extend(n.get("sections", []) or [])
    return arts


def cases(min_refs: int = 3) -> list[ClosureCase]:
    arts = _articles(_load())
    ids = sorted(arts, key=lambda x: int(x))
    text_by = arts
    # cross-reference graph (only edges to articles that exist, excluding self)
    xref = {}
    for a in ids:
        cited = {m for m in _XREF.findall(text_by[a]) if m in text_by and m != a}
        xref[a] = cited
    units = [Unit(id=a, text=text_by[a], tokens=approx_tokens(text_by[a])) for a in ids]
    adjacency = {a: set(xref[a]) for a in ids}
    for a in ids:                                   # make edges bidirectional for structural proximity
        for b in xref[a]:
            adjacency.setdefault(b, set()).add(a)
    out = []
    for a in ids:
        gold = xref[a]
        if len(gold) < min_refs:
            continue
        out.append(ClosureCase(case_id=f"gdpr-art{a}", query=text_by[a][:1500], seed=[a],
                               gold=set(gold), units=units, adjacency=adjacency))
    return out


if __name__ == "__main__":
    cs = cases()
    print(f"financial-regulatory (GDPR): {len(cs)} articles with >=3 cross-references")
    ex = cs[0]
    print(f"  e.g. {ex.case_id}: seed {ex.seed}, gold {sorted(ex.gold, key=int)} ({len(ex.gold)} refs), "
          f"{len(ex.units)} articles in universe")
