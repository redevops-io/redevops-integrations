"""Family 3/4 — basic RAG + context pollution over the LiveRAG anti-parametric corpus (plan §6.B, §7).

Reuses the existing context-vs-model benchmark's data + grader (imported from CR-enterprise), so nothing is
rebuilt: 895 DataMorgana questions grounded in niche FineWeb pages — answers REQUIRE the retrieved context
(a model can't answer from parametric memory, unlike FinanceBench), and the union of all gold docs IS the
corpus, so every other question's docs are a natural distractor pool for the pollution sweep.

Track A (fairness, plan §5): every framework uses the SAME retriever (BM25), the SAME top-K, the SAME
prompt and the SAME model call — only the orchestration wrapper differs, so accuracy/tokens should match
across frameworks (that agreement validates the harness). The model call dominates latency; the framework
delta is the orchestration cost on top. (Dense/hybrid retrieval is a Track-B follow-up, blocked on an embed
endpoint; BM25 is a legitimate shared retriever and is deterministic.)
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import urllib.request
from collections import Counter

from adapters.base import RunConfig, Task, TaskResult

# Reuse the LiveRAG loader + numeric grader from the existing benchmark (no rebuild).
_CVM = "/mnt/backup/projects/CR-enterprise/benchmarks/context-vs-model"
if _CVM not in sys.path:
    sys.path.insert(0, _CVM)
from harness import data as _data      # noqa: E402
from harness import grader as _grader  # noqa: E402

_TOK = re.compile(r"[a-z0-9]+")
_STOP = frozenset("the a an of to in for and or is are was were be on at by with from as this that "
                  "what which who whom where when how it its their his her they them we you can up".split())


def _toks(s: str) -> list[str]:
    return _TOK.findall((s or "").lower())


class BM25:
    """A dependency-free BM25 index over the corpus passages — the shared Track-A retriever."""

    def __init__(self, passages, k1: float = 1.5, b: float = 0.75):
        self.passages = passages
        self.k1, self.b = k1, b
        self.docs = [_toks(p.text) for p in passages]
        self.tf = [Counter(d) for d in self.docs]
        self.dl = [len(d) for d in self.docs]
        self.N = len(self.docs)
        self.avgdl = (sum(self.dl) / self.N) if self.N else 0.0
        df = Counter()
        for d in self.docs:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def search(self, query: str, k: int) -> list:
        q = _toks(query)
        scored = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            for t in q:
                f = tf.get(t)
                if not f:
                    continue
                s += self.idf.get(t, 0.0) * (f * (self.k1 + 1)) / (
                    f + self.k1 * (1 - self.b + self.b * self.dl[i] / (self.avgdl or 1)))
            if s > 0:
                scored.append((s, i))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [self.passages[i] for _, i in scored[:k]]


def build_prompt(question: str, passages: list) -> list[dict]:
    ctx = "\n\n".join(f"[{i + 1}] {p.text}" for i, p in enumerate(passages))
    return [
        {"role": "system", "content": "Answer the question using ONLY the provided context. Be concise "
                                      "(one sentence). If the context does not contain the answer, say you "
                                      "don't know."},
        {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:"},
    ]


def answer_qwen(endpoint: str, model: str, messages: list[dict], max_tokens: int = 96) -> tuple[str, int, int]:
    """One shared model call — identical decoding/thinking-off for every framework (Track A fairness)."""
    body = json.dumps({
        "model": model, "messages": messages, "temperature": 0, "top_p": 1, "seed": 0,
        "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    base = endpoint.rstrip("/")
    url = base + ("/chat/completions" if base.endswith("/v1") else "/v1/chat/completions")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=180))  # noqa: S310 - trusted local inference server
    u = r.get("usage", {})
    return (r["choices"][0]["message"]["content"].strip(),
            int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0)))


def grade(question, text: str) -> bool:
    """Deterministic grade (no LLM judge): numeric match first (reused), else claim-coverage — a gold claim
    is 'covered' if ≥60% of its content words appear in the answer. A defensible proxy for the prose judge
    the source harness uses; flagged as such in the report."""
    if question.is_numeric:
        verdict = _grader.numeric_match(question.answer, text)
        if verdict is not None:
            return bool(verdict)
    low = text.lower()
    for claim in (question.gold_claims or (question.answer,)):
        cw = [w for w in _toks(claim) if w not in _STOP and len(w) > 2]
        if cw and sum(1 for w in cw if w in low) / len(cw) >= 0.6:
            return True
    return False


# ── bindings ──────────────────────────────────────────────────────────────────────────────────────────

class _RagBinding:
    """Base: holds a ref to the workload (shared retriever/corpus) and the endpoint/model from config."""

    def __init__(self, wl: "Workload"):
        self.wl = wl

    def setup(self, config: RunConfig) -> None:
        self.endpoint = config.model_endpoint
        self.model = config.model_id

    def _retrieve(self, question: str) -> list:
        return self.wl.retriever.search(question, self.wl.k)

    def _answer(self, question: str, passages: list) -> tuple[str, int, int]:
        return answer_qwen(self.endpoint, self.model, build_prompt(question, passages))

    def _result(self, task: Task, text: str, it: int, ot: int, ev: list) -> TaskResult:
        return TaskResult(task_id=task.task_id, success=grade(task.payload["q"], text),
                          answer={"text": text[:200]}, evidence_ids=ev,
                          model_calls=[{"input_tokens": it, "output_tokens": ot}])

    def teardown(self) -> None:
        pass


class MinimalRag(_RagBinding):
    def run(self, task: Task) -> TaskResult:
        q = task.payload["q"]
        passages = self._retrieve(q.question)
        text, it, ot = self._answer(q.question, passages)
        return self._result(task, text, it, ot, [p.doc_id for p in passages])


class LangChainRag(_RagBinding):
    def run(self, task: Task) -> TaskResult:
        from langchain_core.runnables import RunnableLambda

        q = task.payload["q"]
        chain = (RunnableLambda(lambda _: self._retrieve(q.question))
                 | RunnableLambda(lambda ps: (ps, self._answer(q.question, ps))))
        passages, (text, it, ot) = chain.invoke(None)
        return self._result(task, text, it, ot, [p.doc_id for p in passages])


class LangGraphRag(_RagBinding):
    def run(self, task: Task) -> TaskResult:
        from typing import TypedDict

        from langgraph.graph import END, START, StateGraph

        q = task.payload["q"]

        class S(TypedDict, total=False):
            passages: list
            text: str
            it: int
            ot: int

        g = StateGraph(S)
        g.add_node("retrieve", lambda s: {"passages": self._retrieve(q.question)})

        def gen(s):
            text, it, ot = self._answer(q.question, s["passages"])
            return {"text": text, "it": it, "ot": ot}

        g.add_node("generate", gen)
        g.add_edge(START, "retrieve")
        g.add_edge("retrieve", "generate")
        g.add_edge("generate", END)
        out = g.compile().invoke({})
        return self._result(task, out["text"], out["it"], out["ot"], [p.doc_id for p in out["passages"]])


class RedevopsRag(_RagBinding):
    """A 2-step ReDevOps mission (retrieve → answer) run via run_program; capabilities do the shared
    retrieve/answer and stash results in a per-run dict."""

    def setup(self, config: RunConfig) -> None:
        super().setup(config)
        from redevops_mission import MissionProgram, Operator, capability, step, template

        self._state: dict = {}

        @template("liverag_rag")
        def _t(mission_id):
            return [step("retrieved", need="retrieve context"),
                    step("answered", need="answer from context", after=["retrieved"])]

        def _do_retrieve(i):
            self._state["passages"] = self._retrieve(self._state["question"])
            return {"n": len(self._state["passages"])}

        def _do_answer(i):
            text, it, ot = self._answer(self._state["question"], self._state["passages"])
            self._state.update(text=text, it=it, ot=ot)
            return {"ok": True}

        ops = [Operator("rag-worker", [
            capability("rag.retrieve", handler=_do_retrieve, provides=["retrieved"]),
            capability("rag.answer", handler=_do_answer, provides=["answered"]),
        ])]
        self._program = MissionProgram.from_template("liverag_rag", goal="answer a question", grants=[])
        self._operators = ops

    def run(self, task: Task) -> TaskResult:
        from redevops_mission import run_program

        q = task.payload["q"]
        self._state = {"question": q.question}
        run_program(self._program, self._operators)
        return self._result(task, self._state.get("text", ""), self._state.get("it", 0),
                            self._state.get("ot", 0), [p.doc_id for p in self._state.get("passages", [])])


class Workload:
    name = "rag/liverag"

    def __init__(self, n: int = 50, k: int = 5):
        self.k = k
        self._questions = _data.load_questions()[:n]
        self.retriever = BM25(_data.load_corpus().passages)
        self.minimal_binding = MinimalRag(self)
        self.langchain_binding = LangChainRag(self)
        self.langgraph_binding = LangGraphRag(self)
        self.redevops_binding = RedevopsRag(self)

    def tasks(self) -> list[Task]:
        return [Task(task_id=q.id, payload={"q": q},
                     expected={"answer": q.answer, "numeric": q.is_numeric}) for q in self._questions]
