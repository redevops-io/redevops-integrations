"""LangGraph adapter (Baseline B).

Each workload is expressed as an equivalent LangGraph: state graph + nodes + tools + conditional edges,
with LangGraph's own checkpointing for the replay/recovery family. In Track A it must use the SAME model
endpoint, retriever, top-K and tool outputs as every other adapter (fairness, plan §5); in Track B it may
use LangGraph's native memory/state/checkpointer/tool-routing (plan §1, Track B).

LangGraph is a NET-NEW dependency here (the ReDevOps stack does not depend on it): `pip install -e
'.[langgraph]'`. Nothing reusable exists — this adapter is built from scratch per workload family.

STATUS: stub. Each workload exposes a `langgraph_binding` that builds the graph for that task shape; wire
Family 1/2 (deterministic node graphs + concurrency) and Family 7 (checkpointer) first — they exercise the
framework overhead / scheduling / recovery this benchmark exists to measure.
"""
from __future__ import annotations

from adapters.base import BenchmarkAdapter, RunConfig, Task, TaskResult


class LanggraphAdapter(BenchmarkAdapter):
    name = "langgraph"

    def setup(self, workload, config: RunConfig) -> None:
        self.workload = workload
        self.config = config
        self._binding = getattr(workload, "langgraph_binding", None)
        if self._binding is None:
            raise NotImplementedError(
                f"workload {getattr(workload, 'name', workload)!r} has no langgraph_binding yet")
        self._binding.setup(config)  # compiles the StateGraph with a checkpointer for Family 7

    def run(self, task: Task) -> TaskResult:
        return self._binding.run(task)

    def checkpoint(self) -> bytes:
        return self._binding.checkpoint()   # LangGraph checkpointer snapshot

    def recover(self, state: bytes) -> None:
        self._binding.recover(state)

    def teardown(self) -> None:
        if self._binding is not None:
            self._binding.teardown()
        self.workload = None
