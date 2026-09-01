"""Baseline C — minimal custom Python orchestration (plan §1, §14).

This is the control that separates *framework overhead* from *model/tool time*: a hand-written loop that
calls the shared inference endpoint and the workload's tools directly, with no orchestration framework. Any
latency/RAM a framework adds shows up as its delta over this baseline. It is intentionally the simplest
correct implementation of each workload's task, nothing more.
"""
from __future__ import annotations

import json
import urllib.request

from adapters.base import BenchmarkAdapter, RunConfig, Task, TaskResult


class MinimalAdapter(BenchmarkAdapter):
    name = "minimal"

    def setup(self, workload, config: RunConfig) -> None:
        self.workload = workload
        self.config = config
        # Uniform with the other adapters: a workload may provide a `minimal_binding` (e.g. a no-model
        # orchestration graph). Model workloads instead expose build_messages/parse_answer/grade (below).
        self._binding = getattr(workload, "minimal_binding", None)
        if self._binding is not None:
            self._binding.setup(config)

    def _chat(self, messages: list[dict]) -> dict:
        """One OpenAI-compatible completion against the shared endpoint. Returns the raw response so the
        runner can read `usage` for exact token accounting (never self-report tokens)."""
        payload = {"model": self.config.model_id, "messages": messages, **self.config.decoding}
        # Qwen3/Nemotron reasoning models default to thinking mode; disable it for benchmark determinism via
        # chat_template_kwargs (vLLM passes it to the chat template). Set config.extra["chat_template_kwargs"].
        ctk = self.config.extra.get("chat_template_kwargs")
        if ctk:
            payload["chat_template_kwargs"] = ctk
        body = json.dumps(payload).encode()
        # model_endpoint is an OpenAI-style base URL already ending in /v1 (see configs/deterministic.yaml).
        base = self.config.model_endpoint.rstrip("/")
        url = base + ("/chat/completions" if base.endswith("/v1") else "/v1/chat/completions")
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:  # noqa: S310 - trusted local inference server
            return json.loads(r.read())

    def run(self, task: Task) -> TaskResult:
        if self._binding is not None:
            return self._binding.run(task)
        # Model workloads: the minimal loop wires build_messages → one model call → grade.
        messages = self.workload.build_messages(task)
        resp = self._chat(messages)
        answer = self.workload.parse_answer(resp)
        usage = resp.get("usage", {})
        return TaskResult(
            task_id=task.task_id,
            success=self.workload.grade(task, answer),
            answer=answer,
            model_calls=[{
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            }],
        )

    def teardown(self) -> None:
        if self._binding is not None:
            self._binding.teardown()
        self.workload = None
