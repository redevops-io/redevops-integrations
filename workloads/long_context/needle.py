"""Family 5 — long-context / evidence resurrection (plan §7). Needle-in-a-real-haystack.

A synthetic needle ("the secure access code for facility {KEY} is {VALUE}", random KEY/VALUE so it's in no
training set) is buried at 90% depth in a haystack of real LiveRAG passages sized 10k…100k tokens. The
model's window is ~40k (Qwen 40960). Two arms answer "what is the code for facility {KEY}?":

  - **full** — stuff the haystack, truncated to the window (keep the earliest 40k tokens). Once the haystack
    exceeds ~44k, the 90%-depth needle is truncated OUT → the answer isn't in the prompt → collapse.
  - **select** — a tiny BM25 lookup on the (unique) KEY retrieves the needle passage → ~1k-token context →
    answers at every horizon.

This is the "a context window is a cliff, not a ramp" result: past the window, dumping everything in falls
to zero while managed context holds full accuracy at a fraction of the tokens (plan §6.A, the
answers-from-less-context finding), reproduced through the benchmark harness on the local 27B model.
"""
from __future__ import annotations

import random

from adapters.base import RunConfig, Task, TaskResult
from workloads.rag import liverag as lr

HORIZONS = [10_000, 25_000, 50_000, 100_000]   # target haystack tokens
WINDOW = 36_000                                 # truncate here — headroom under Qwen's 40960 hard limit
_CHARS_PER_TOK = 4
_DEPTH = 0.9                                     # needle position within the haystack


def _needle_text(key: str, value: str) -> str:
    return f"Facility access record. The secure access code for facility {key} is {value}. End of record."


class Workload:
    name = "long_context/needle"

    def __init__(self, arm: str, n: int = 8, seed: int = 7):
        self._passages = lr._data.load_corpus().passages
        self.n = n
        self._by_h: dict[int, list[str]] = {}
        rng = random.Random(seed)
        self._needles = []
        for i in range(n):
            key = "GX-" + "".join(rng.choice("0123456789") for _ in range(5))
            value = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(6))
            self._needles.append((key, value))
        self.minimal_binding = {"full": FullArm, "select": SelectArm}[arm](self)

    def haystack(self, horizon: int) -> list[str]:
        """A list of real-passage texts totaling ~`horizon` tokens (cached per horizon)."""
        if horizon not in self._by_h:
            budget = horizon * _CHARS_PER_TOK
            out, total = [], 0
            for p in self._passages:
                out.append(p.text)
                total += len(p.text)
                if total >= budget:
                    break
            self._by_h[horizon] = out
        return self._by_h[horizon]

    def tasks(self) -> list[Task]:
        tasks = []
        for h in HORIZONS:
            for i, (key, value) in enumerate(self._needles):
                tasks.append(Task(task_id=f"h{h}-{i}", payload={"h": h, "key": key, "value": value},
                                  expected={"value": value}))
        return tasks


class _Arm:
    def __init__(self, wl: Workload):
        self.wl = wl

    def setup(self, config: RunConfig) -> None:
        self.endpoint, self.model = config.model_endpoint, config.model_id

    def _messages(self, key: str, context: str) -> list[dict]:
        return [
            {"role": "system", "content": "Answer ONLY from the provided context. Give just the code."},
            {"role": "user", "content": f"Context:\n{context}\n\nWhat is the secure access code for "
                                        f"facility {key}? Answer with the code only."},
        ]

    def _answer(self, key: str, context: str):
        return lr.answer_qwen(self.endpoint, self.model, self._messages(key, context), max_tokens=16)

    def _result(self, task, text, it, ot):
        return TaskResult(task_id=task.task_id,
                          success=task.expected["value"].lower() in text.lower(),
                          answer={"text": text[:40]}, model_calls=[{"input_tokens": it, "output_tokens": ot}],
                          extra={"horizon": task.payload["h"]})

    def teardown(self) -> None:
        pass


class FullArm(_Arm):
    def run(self, task: Task) -> TaskResult:
        h, key, value = task.payload["h"], task.payload["key"], task.payload["value"]
        passages = list(self.wl.haystack(h))
        passages.insert(int(_DEPTH * len(passages)), _needle_text(key, value))
        context = "\n\n".join(passages)[: WINDOW * _CHARS_PER_TOK]   # truncate to the model window
        text, it, ot = self._answer(key, context)
        return self._result(task, text, it, ot)


class SelectArm(_Arm):
    def run(self, task: Task) -> TaskResult:
        h, key, value = task.payload["h"], task.payload["key"], task.payload["value"]
        passages = list(self.wl.haystack(h))
        needle = _needle_text(key, value)
        passages.insert(int(_DEPTH * len(passages)), needle)
        # a small BM25 over just this haystack; the unique KEY ranks the needle first (managed context).
        idx = lr.BM25([_P(t, i) for i, t in enumerate(passages)])
        top = idx.search(key, 3)
        context = "\n\n".join(p.text for p in top)
        text, it, ot = self._answer(key, context)
        return self._result(task, text, it, ot)


class _P:
    """Minimal passage shim (BM25 needs .text)."""

    __slots__ = ("text", "doc_id", "id")

    def __init__(self, text: str, i: int):
        self.text = text
        self.doc_id = str(i)
        self.id = str(i)
