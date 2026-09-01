"""The common adapter every framework implements (plan §14).

One BenchmarkAdapter per stack — redevops / langchain / langgraph / minimal — so a single runner drives
all of them over the same workload with the same telemetry. The adapter is deliberately thin: it wires a
stack to a workload and executes one task; all measurement (timing, memory, GPU, tokens) is done *around*
these calls by the runner, so the framework under test never reports its own numbers.

Fairness invariant (Track A, plan §5): every adapter receives the SAME semantic task and equivalent
capabilities — same model endpoint, prompts, tool outputs, retriever, top-K, dataset snapshot, and
correctness rubric. Track B relaxes this so each stack may use its native architecture; the `track` on the
RunConfig records which regime a result belongs to.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RunConfig:
    """Everything that must be pinned for a result to be reproducible (feeds the §15 manifest)."""

    workload: str                     # e.g. "rag/context_pollution"
    track: str                        # "A" (orchestration-controlled) | "B" (native-architecture)
    model_endpoint: str               # OpenAI-compatible base URL of the shared inference server
    model_id: str
    tokenizer_hash: str = ""
    quantization: str = ""            # e.g. "Q4_K_M" / "AWQ" / "gptq-int4"
    decoding: dict[str, Any] = field(default_factory=lambda: {"temperature": 0, "top_p": 1, "seed": 0})
    retriever: str = ""               # shared retriever id (Track A) or "native" (Track B)
    top_k: int = 0
    dataset_id: str = ""
    dataset_hash: str = ""
    as_of: str | None = None          # point-in-time context where the workload is temporal
    known_at: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    """One benchmark item. `payload` is workload-specific; `expected` feeds deterministic grading (§11)."""

    task_id: str
    payload: dict[str, Any]
    expected: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """What an adapter returns for one task. The runner enriches this with timing/memory/gpu it measured
    externally, then serializes to the common result schema (metrics/schema.py). Adapters MUST NOT report
    their own wall-clock/memory — only what they genuinely know (the answer, evidence used, and the
    model/tool calls they made, for token accounting)."""

    task_id: str
    success: bool                                  # graded by the workload's rubric, not self-reported
    answer: dict[str, Any] = field(default_factory=dict)
    evidence_ids: list[str] = field(default_factory=list)
    model_calls: list[dict[str, Any]] = field(default_factory=list)  # per call: input/output tokens, etc.
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class BenchmarkAdapter(abc.ABC):
    """Implemented once per stack. Lifecycle: setup → (run|checkpoint|recover)* → teardown."""

    name: str = "base"

    @abc.abstractmethod
    def setup(self, workload: Any, config: RunConfig) -> None:
        """Build the stack for this workload+config (wire the shared model endpoint, retriever, tools)."""

    @abc.abstractmethod
    def run(self, task: Task) -> TaskResult:
        """Execute ONE task and return its result. Called inside the runner's measurement window."""

    def checkpoint(self) -> bytes:
        """Serialize resumable state (Family 7 replay/recovery). Default: unsupported."""
        raise NotImplementedError(f"{self.name}: checkpoint not supported")

    def recover(self, state: bytes) -> None:
        """Restore from a checkpoint (Family 7). Default: unsupported."""
        raise NotImplementedError(f"{self.name}: recover not supported")

    def teardown(self) -> None:
        """Release resources so between-run memory is measured cleanly (§9)."""
