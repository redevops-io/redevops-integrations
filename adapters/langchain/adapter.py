"""LangChain adapter (Baseline A).

Each workload is expressed with LangChain primitives (LCEL chains / AgentExecutor / retrievers / tools /
memory). Track A: same model endpoint, retriever, top-K, tool outputs as every other adapter. Track B:
LangChain's native memory/retrievers/agent patterns.

LangChain is a NET-NEW dependency here: `pip install -e '.[langchain]'` (langchain + langchain-openai;
point ChatOpenAI at the shared vLLM `/v1` endpoint). Nothing reusable exists — built per workload family.

STATUS: stub. Wire the RAG (Family 3/4) and multi-step agent (Family 12) shapes first — LangChain's
AgentExecutor is the natural Baseline-A counterpart to a ReDevOps mission.
"""
from __future__ import annotations

from adapters.base import BenchmarkAdapter, RunConfig, Task, TaskResult


class LangchainAdapter(BenchmarkAdapter):
    name = "langchain"

    def setup(self, workload, config: RunConfig) -> None:
        self.workload = workload
        self.config = config
        self._binding = getattr(workload, "langchain_binding", None)
        if self._binding is None:
            raise NotImplementedError(
                f"workload {getattr(workload, 'name', workload)!r} has no langchain_binding yet")
        self._binding.setup(config)

    def run(self, task: Task) -> TaskResult:
        return self._binding.run(task)

    def teardown(self) -> None:
        if self._binding is not None:
            self._binding.teardown()
        self.workload = None
