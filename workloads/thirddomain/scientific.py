"""Scientific-citation domain — SciFact (real, human-labeled claim→evidence closures).

A claim cites document(s); within the cited abstracts, a human-annotated set of sentences forms the evidence
that supports or refutes the claim. That evidence set is the closure: the reviewer of the claim must consider all
of it. Units = sentences of the cited abstracts; seed = the first evidence sentence; gold = all evidence
sentences; structural edges = intra-abstract sentence adjacency (evidence spans cluster) — the citation already
scopes the retrieval universe to the cited docs. Semantic-leaning closure (which sentences are evidence is a
meaning judgement), a real second point near the legal end of the axis.

Data: SciFact public release (allenai). Auto-downloaded to data/scifact/ if absent (the datasets *loader* is
deprecated; the raw JSONL is not).
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
import urllib.request

from common import ClosureCase, Unit, approx_tokens

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "data", "scifact", "data")
URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"


def _ensure():
    if os.path.exists(os.path.join(ROOT, "corpus.jsonl")):
        return
    os.makedirs(os.path.dirname(os.path.dirname(ROOT)), exist_ok=True)
    data = urllib.request.urlopen(URL, timeout=90).read()
    tarfile.open(fileobj=io.BytesIO(data)).extractall(os.path.join(HERE, "data", "scifact"), filter="data")


def cases(min_evidence: int = 3, splits=("claims_train", "claims_dev"), distractors: int = 10) -> list[ClosureCase]:
    _ensure()
    corpus = {}
    for line in open(os.path.join(ROOT, "corpus.jsonl")):
        d = json.loads(line)
        corpus[int(d["doc_id"])] = d["abstract"]        # list[str] sentences
    all_ids = sorted(corpus)                              # deterministic distractor pool
    out: list[ClosureCase] = []
    for split in splits:
        path = os.path.join(ROOT, f"{split}.jsonl")
        if not os.path.exists(path):
            continue
        for line in open(path):
            c = json.loads(line)
            ev = c.get("evidence") or {}
            if not ev:
                continue
            # gold evidence sentences across cited docs; skip if a cited doc is missing
            gold, units, adjacency = set(), [], {}
            ok = True
            ev_pairs = []
            for did, groups in ev.items():
                did_i = int(did)
                if did_i not in corpus:
                    ok = False
                    break
                sents = corpus[did_i]
                ids = [f"{did_i}:{i}" for i in range(len(sents))]
                for i, s in enumerate(sents):
                    units.append(Unit(id=ids[i], text=s, tokens=approx_tokens(s)))
                for a, b in zip(ids, ids[1:]):            # intra-abstract sentence adjacency (structural)
                    adjacency.setdefault(a, set()).add(b)
                    adjacency.setdefault(b, set()).add(a)
                for g in groups:
                    for si in g["sentences"]:
                        ev_pairs.append((did_i, si))
            if not ok:
                continue
            gold = {f"{d}:{i}" for d, i in ev_pairs}
            if len(gold) < min_evidence:
                continue
            # deterministic distractor docs (not cited) → a realistic retrieval universe with noise
            cited = {int(did) for did in ev}
            start = int(hashlib.md5(str(c["id"]).encode()).hexdigest(), 16) % max(1, len(all_ids))
            picked = 0
            for off in range(len(all_ids)):
                did_i = all_ids[(start + off) % len(all_ids)]
                if did_i in cited:
                    continue
                sents = corpus[did_i]
                ids = [f"{did_i}:{i}" for i in range(len(sents))]
                for i, s in enumerate(sents):
                    units.append(Unit(id=ids[i], text=s, tokens=approx_tokens(s)))
                for a, b in zip(ids, ids[1:]):
                    adjacency.setdefault(a, set()).add(b)
                    adjacency.setdefault(b, set()).add(a)
                picked += 1
                if picked >= distractors:
                    break
            seed = sorted(gold, key=lambda x: (int(x.split(":")[0]), int(x.split(":")[1])))[:1]
            out.append(ClosureCase(case_id=f"scifact-{c['id']}", query=c["claim"], seed=seed,
                                   gold=gold, units=units, adjacency=adjacency))
    return out


if __name__ == "__main__":
    cs = cases()
    print(f"scientific-citation (SciFact): {len(cs)} cases with >=3 evidence sentences")
    ex = cs[0]
    print(f"  e.g. {ex.case_id}: {len(ex.units)} units, gold {len(ex.gold)}, seed {ex.seed}")
    print(f"       claim: {ex.query[:90]}")
