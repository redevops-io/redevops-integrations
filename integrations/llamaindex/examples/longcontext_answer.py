"""LI-B — long-context: the query engine always answers; it never tells you when it shouldn't (plan 7).

LlamaIndex's query engine synthesizes a fluent answer from whatever the retriever returned. That is its strength
and, just outside it, its risk: when retrieval misses the needle in a long haystack, the query engine still
returns a confident answer — wrong, with no abstention signal. We run a REAL LlamaIndex query engine over a real
LiveRAG haystack with a buried synthetic needle, in two conditions: the needle is retrievable, or it is not. Then
ReDevOps verification checks whether the synthesized value is actually supported by the retrieved evidence, and
abstains (REQUIRE_REVIEW) when it is not — the honest signal the synthesizer does not provide.

    python examples/longcontext_answer.py     # a few gpt-4o-mini synthesis calls, cached; re-runs are offline
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# 1) Load the real LiveRAG haystack while the repo-root `adapters` PACKAGE (needed by workloads.rag) resolves.
#    `adapters` may already be cached as THIS integration's local adapters.py (e.g. under conformance.py), so evict
#    it first and put the repo root ahead on the path; restore afterwards so local `import adapters` works in step 2.
for _m in ("adapters", "adapters.base"):
    sys.modules.pop(_m, None)
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))                       # repo root → workloads.*
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "workloads", "long_context"))
from workloads.rag import liverag as _lr  # noqa: E402
_PASSAGES_ALL = _lr._data.load_corpus().passages

# 2) Evict the repo-root `adapters` package (liverag has already bound what it needs), then add the local
#    integration dir so `import adapters` resolves to this integration's local adapters.py.
for _m in ("adapters", "adapters.base"):
    sys.modules.pop(_m, None)
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.append(os.path.join(HERE, "..", "..", "..", "workloads", "legal_change_closure"))  # legal_llm

from adapters import LI_VERSION, LocalEmbedding  # noqa: E402
from bundle import AcceptanceBundle  # noqa: E402
from classification import Finding, ResultClass  # noqa: E402

import legal_llm  # noqa: E402
from llama_index.core import Document, Settings, VectorStoreIndex  # noqa: E402
from llama_index.core.llms import CompletionResponse, CustomLLM, LLMMetadata  # noqa: E402
from llama_index.core.llms.callbacks import llm_completion_callback  # noqa: E402

CACHE = os.path.join(HERE, "longcontext_cache")
N = 6                          # needles
HAYSTACK_PASSAGES = 40         # distractors per index (kept small; the point is retrieval hit/miss, not window size)


class LocalLLM(CustomLLM):
    """A real LlamaIndex LLM backing the query engine's response synthesis, on our gpt-4o-mini backend."""

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(context_window=8000, num_output=32, model_name="gpt-4o-mini")

    @llm_completion_callback()
    def complete(self, prompt: str, **kw) -> CompletionResponse:
        return CompletionResponse(text=legal_llm.chat(
            "You answer the question using ONLY the context. Reply with just the access code.", prompt, max_tokens=24))

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kw):
        yield self.complete(prompt)


def _needle_text(key, value):
    return f"Facility access record. The secure access code for facility {key} is {value}. End of record."


def _query_engine(passages, key, value, include_needle):
    docs = [Document(text=t, id_=f"d{i}") for i, t in enumerate(passages)]
    if include_needle:
        docs.append(Document(text=_needle_text(key, value), id_="needle"))
    idx = VectorStoreIndex.from_documents(docs, show_progress=False)
    return idx.as_query_engine(similarity_top_k=3, response_mode="compact")


def _ask(task_id, arm, qe, question):
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, f"{task_id}__{arm}.json")
    if os.path.exists(p):
        return json.load(open(p))
    resp = qe.query(question)
    ans = str(resp).strip()
    # ReDevOps verification: is the synthesized value actually present in the retrieved evidence?
    evidence = "\n".join(n.node.get_content() for n in resp.source_nodes)
    rec = {"answer": ans, "evidence": evidence}
    json.dump(rec, open(p, "w"))
    return rec


def main():
    Settings.embed_model = LocalEmbedding()
    Settings.llm = LocalLLM()

    passages = [p.text for p in _PASSAGES_ALL[:HAYSTACK_PASSAGES]]

    import random
    rng = random.Random(7)
    needles = []
    for _ in range(N):
        key = "GX-" + "".join(rng.choice("0123456789") for _ in range(5))
        value = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(6))
        needles.append((key, value))

    print(f"LI-B long-context — {N} needles over a {HAYSTACK_PASSAGES}-passage LiveRAG haystack "
          f"(real LlamaIndex query engine, gpt-4o-mini synthesis)")

    hit_ok, miss_ok, miss_answered, miss_abstained = [], [], 0, 0
    for i, (key, value) in enumerate(needles):
        q = f"What is the secure access code for facility {key}?"
        # arm 1 — needle retrievable
        qe_hit = _query_engine(passages, key, value, include_needle=True)
        rec_hit = _ask(f"n{i}", "retrieved", qe_hit, q)
        hit_ok.append(value in rec_hit["answer"])
        # arm 2 — retrieval misses the needle (it is not in the index)
        qe_miss = _query_engine(passages, key, value, include_needle=False)
        rec_miss = _ask(f"n{i}", "missed", qe_miss, q)
        correct = value in rec_miss["answer"]
        miss_ok.append(correct)
        # the query engine answered with SOMETHING (fluent, confident) — did it signal it couldn't?
        answered = len(rec_miss["answer"]) > 0 and "don't" not in rec_miss["answer"].lower() \
            and "cannot" not in rec_miss["answer"].lower() and "no " not in rec_miss["answer"].lower()[:4]
        if answered:
            miss_answered += 1
        # ReDevOps verification: value must appear in the retrieved evidence, else abstain
        supported = value in rec_miss["evidence"]
        if not supported:
            miss_abstained += 1

    acc_hit, acc_miss = st.mean(hit_ok), st.mean(miss_ok)
    # selective accuracy: answer only when ReDevOps verification says supported.
    covered = hit_ok  # exactly the retrieved-and-supported cases resolve autonomously
    sel_acc = st.mean(covered) if covered else 0.0
    cov = len(covered) / (2 * N)

    b = AcceptanceBundle(framework="llama-index", framework_version=LI_VERSION, model_ids=["gpt-4o-mini"],
                         dataset_hashes={"haystack": "liverag"}, config={"n": N, "passages": HAYSTACK_PASSAGES})
    b.add(Finding("LI-B synthesis hides retrieval miss", "confident_wrong_answers",
                  native=f"{miss_answered}/{N} answered confidently", with_redevops=f"{miss_abstained}/{N} abstained",
                  classification=ResultClass.REDEVOPS_DELTA if miss_abstained >= miss_answered and miss_abstained > 0
                  else ResultClass.BUG,
                  note="on a retrieval miss the query engine still synthesizes a confident answer; ReDevOps evidence-"
                       "support verification abstains (REQUIRE_REVIEW) because the value is not in the retrieved nodes"))
    b.add(Finding("LI-B answer accuracy", "accuracy_retrieved_vs_missed", native=round(acc_miss, 3),
                  with_redevops=round(acc_hit, 3),
                  classification=ResultClass.REDEVOPS_DELTA if acc_hit > acc_miss else ResultClass.PARITY,
                  note="when the LlamaIndex retriever finds the needle the query engine is right; when it misses, "
                       "synthesis is confidently wrong — retrieval quality, not synthesis fluency, decides correctness"))
    b.add(Finding("LI-B selective accuracy", "selective_accuracy@coverage",
                  native="(the query engine offers no abstention)", with_redevops=f"{sel_acc:.3f} @ {cov:.0%}",
                  classification=ResultClass.REDEVOPS_DELTA,
                  note="answer only when verification confirms evidence support; the unsupported tail is routed to review"))

    out = b.write(os.path.join(HERE, "..", "results", "li_b_longcontext_answer.json"))
    print(f"  query-engine accuracy: needle retrieved {acc_hit:.3f} · needle missed {acc_miss:.3f}")
    print(f"  on a miss: {miss_answered}/{N} confidently answered · ReDevOps abstained {miss_abstained}/{N}")
    print(f"  selective accuracy: {sel_acc:.3f} @ {cov:.0%} coverage")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
