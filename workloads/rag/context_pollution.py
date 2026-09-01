"""Family 4 — context pollution (plan §7). The headline "better context beats more context" test.

Fix a K-passage context budget; at pollution ratio r, r·K of the passages are irrelevant distractors
(random corpus passages) and (1−r)·K are the question's true top-BM25 passages. The SAME polluted pool is
built for both arms (deterministic seed), so the only difference is the context strategy:

  - **naive** — stuff all K polluted passages into the prompt (what a plain RAG chain does).
  - **select** — a runtime context-selection step: re-score the K passages by query relevance and keep only
    the relevant ones (distractors, being random, score ~0 and drop out). This models ReDevOps Context
    Runtime selection; `context_runtime.ContextRuntime` is the production selector — this is a defensible
    deterministic stand-in so the family runs without that dependency.

Expected (plan §6.B): naive accuracy degrades as r rises (noise crowds out the answer) while select holds,
and select sends far fewer tokens — pronounced on the local 27B model.
"""
from __future__ import annotations

import random
from collections import Counter

from adapters.base import RunConfig, Task, TaskResult
from workloads.rag import liverag as lr

RATIOS = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95]


def _rel_score(retriever: lr.BM25, passage, qterms: set[str]) -> float:
    tf = Counter(lr._toks(passage.text))
    return sum(retriever.idf.get(t, 0.0) for t in qterms if t in tf)


def polluted_context(retriever: lr.BM25, question, k: int, r: float):
    """K passages with r·K random distractors + (1−r)·K true signal, deterministically shuffled."""
    signal = retriever.search(question.question, k)
    n_dist = int(round(r * k))
    n_sig = k - n_dist
    rng = random.Random(hash(question.id) & 0xFFFFFFFF)
    pool = retriever.passages
    sig_ids = {p.id for p in signal}
    distractors, seen = [], set()
    while len(distractors) < n_dist and len(seen) < len(pool):
        p = pool[rng.randrange(len(pool))]
        if p.id in sig_ids or p.id in seen:
            seen.add(p.id)
            continue
        seen.add(p.id)
        distractors.append(p)
    ctx = signal[:n_sig] + distractors
    rng.shuffle(ctx)
    return ctx


class _Arm:
    def __init__(self, wl: "Workload"):
        self.wl = wl

    def setup(self, config: RunConfig) -> None:
        self.endpoint, self.model = config.model_endpoint, config.model_id

    def _answer(self, question: str, passages: list):
        return lr.answer_qwen(self.endpoint, self.model, lr.build_prompt(question, passages))

    def _result(self, task, text, it, ot, passages):
        return TaskResult(task_id=task.task_id, success=lr.grade(task.payload["q"], text),
                          answer={"text": text[:160]}, model_calls=[{"input_tokens": it, "output_tokens": ot}],
                          extra={"ratio": task.payload["r"], "ctx_passages": len(passages)})

    def teardown(self) -> None:
        pass


class NaiveArm(_Arm):
    def run(self, task: Task) -> TaskResult:
        q, r = task.payload["q"], task.payload["r"]
        ctx = polluted_context(self.wl.retriever, q, self.wl.k, r)
        text, it, ot = self._answer(q.question, ctx)
        return self._result(task, text, it, ot, ctx)


class SelectArm(_Arm):
    def run(self, task: Task) -> TaskResult:
        q, r = task.payload["q"], task.payload["r"]
        ctx = polluted_context(self.wl.retriever, q, self.wl.k, r)
        qterms = set(lr._toks(q.question))
        scored = [(p, _rel_score(self.wl.retriever, p, qterms)) for p in ctx]
        top = max((s for _, s in scored), default=0.0)
        # keep the query-relevant passages (distractors score ~0); always keep at least the single best.
        kept = [p for p, s in scored if s >= 0.3 * top and s > 0] or [max(scored, key=lambda x: x[1])[0]]
        text, it, ot = self._answer(q.question, kept)
        return self._result(task, text, it, ot, kept)


class Workload:
    name = "rag/context_pollution"

    def __init__(self, retriever: lr.BM25, questions: list, k: int, arm: str):
        self.retriever, self._questions, self.k = retriever, questions, k
        self.minimal_binding = {"naive": NaiveArm, "select": SelectArm}[arm](self)

    def tasks(self) -> list[Task]:
        return [Task(task_id=f"r{r}-{q.id}", payload={"q": q, "r": r},
                     expected={"answer": q.answer}) for r in RATIOS for q in self._questions]
