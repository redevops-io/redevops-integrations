"""ReDevOps Runtime Stack adapter.

Drives the stack through its SANCTIONED public boundaries (never `agentic_os.mission.*` directly):

  - Mission execution / orchestration (Families 1, 2, 12) → `redevops_mission.run_program(program, operators)`
    over a `MissionProgram` authored with `template`/`step`/`capability`/`Operator`.
  - Checkpoint / replay / recovery (Family 7) → `export_bundle` + `replay_bundle` / `verify_bundle`.
  - Retrieval (Families 3, 4, 8) → `redevops_rag.RAG(...).search()` / `hybrid_search` / `diver_search`.
  - Context selection / "what the model sees" (Families 4, 5) → `context_runtime.ContextRuntime.default(docs)
    .run(goal, as_of=...)` and `.explain()`.
  - Incremental recomputation (Family 6) → `discovery_runtime.incremental` (DiscoveryCheckpoint,
    discover_incremental, mark_stale / mark_invalidated).
  - Intent → mission compilation (Family 11) → `discovery_runtime.DiscoveryRuntime.draft/clarifications/
    resolve/seal`.

Install (from sibling checkouts): `pip install -e /mnt/backup/projects/mission-sdk -e
/mnt/backup/projects/redevops-rag -e /mnt/backup/projects/contextos`. discovery-runtime is Proprietary —
gate the Family 6/11 arms behind an opt-in so the public benchmark core stays runnable without it.

STATUS: skeleton. Each workload provides a `redevops_binding` describing which surface it exercises; this
adapter dispatches to it. Fill per-family in wave 1 (see docs/PLAN.md §16 headline six).
"""
from __future__ import annotations

from adapters.base import BenchmarkAdapter, RunConfig, Task, TaskResult


class RedevopsAdapter(BenchmarkAdapter):
    name = "redevops"

    def setup(self, workload, config: RunConfig) -> None:
        self.workload = workload
        self.config = config
        # Point the stack's model + retriever at the SAME shared endpoint/snapshot the other adapters use
        # (Track A fairness). Track B lets the workload's redevops_binding enable native context selection,
        # routing, hybrid retrieval, etc.
        self._binding = getattr(workload, "redevops_binding", None)
        if self._binding is None:
            raise NotImplementedError(
                f"workload {getattr(workload, 'name', workload)!r} has no redevops_binding yet")
        self._binding.setup(config)

    def run(self, task: Task) -> TaskResult:
        # The binding returns a normalized (success, answer, evidence_ids, model_calls, tool_calls) tuple by
        # driving run_program / RAG / ContextRuntime / discovery as appropriate for its family.
        return self._binding.run(task)

    def checkpoint(self) -> bytes:
        # Family 7: export the mission's self-verifying case bundle.
        return self._binding.export_bundle()

    def recover(self, state: bytes) -> None:
        # Family 7: replay the sealed bundle; the runner asserts replay equivalence + zero duplicate effects.
        self._binding.replay_bundle(state)

    def teardown(self) -> None:
        if self._binding is not None:
            self._binding.teardown()
        self.workload = None
