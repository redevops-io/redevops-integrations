"""Resolution loop (build step 6 / P4): does an arm's retrieved context let the agent actually MAKE the
change — every closure member, edited in the right place — such that a frozen acceptance test passes?

For each arm we: create an isolated git worktree of `agentic-os@f8ce79c`; give the agent ONLY that arm's
retrieved context (symbol source + file + line range) plus the task; ask for edits as SEARCH/REPLACE blocks;
apply them; drop in the family's frozen acceptance test; run pytest (acceptance targets + `must_not_regress`).
`resolved` = every target passes. This is the "finding the file isn't enough" measurement — an arm that
misses the seed definition or a contract caller should fail application/tests even if its recall looked ok.

Agent model: Qwen on the RTX PRO 4500 Blackwell 32GB (`http://192.168.40.105:30807/v1`), thinking DISABLED
(reasoning tokens otherwise wreck both latency and the edit format — the LangChain/LangGraph harness hit the
same wall). Model id is read live from `/v1/models`. No network beyond that endpoint.

    .venv/bin/python workloads/code_retrieval/apply_loop.py --task signature_change_v1 --budget 4000
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.request

import yaml

from resolver import build_call_graph, index_repo
from retrievers import (existing_cr_arm, existing_cr_hybrid_arm, graph_closure_arm,
                        graph_plus_hybrid_arm, naive_semantic_arm, oracle_closure_arm)
from scoring import Oracle, RetrievedContext, RetrievedUnit

HERE = os.path.dirname(os.path.abspath(__file__))
# Agent backend. `local` = Qwen3.8-27B on the proxmox GPU (vLLM, OpenAI-compatible, thinking-disable kwarg).
# `kimi` = Moonshot Kimi over the cloud API (bearer auth, no vLLM kwarg). Configured in main() via --provider.
ENDPOINT = os.environ.get("CR_MODEL_ENDPOINT", "http://192.168.40.105:31111/v1")
_API_KEY = ""            # bearer token (cloud providers); empty = local, no auth header
_MODEL_NAME = ""         # explicit model id; empty = discover via /v1/models
_VLLM_KWARGS = True      # send chat_template_kwargs (vLLM-only; Moonshot rejects it)
_TEMPERATURE = 0.0       # kimi-k2.7-code is a reasoning model that ONLY accepts temperature=1
VENV_PY = os.environ.get("CR_VENV_PY", "/mnt/backup/projects/redevops-benchmarks/.venv/bin/python")
# Reasoning models (kimi-k2.7-code, Qwen3.8 with thinking) spend output budget on reasoning before the edit
# blocks; a small cap starves the actual answer to empty. Give edit generations real headroom.
EDIT_MAX_TOKENS = int(os.environ.get("CR_EDIT_MAX_TOKENS", "8192"))


def _configure(provider: str) -> None:
    global ENDPOINT, _API_KEY, _MODEL_NAME, _VLLM_KWARGS, _TEMPERATURE
    if provider == "kimi":
        ENDPOINT = (os.environ.get("KIMI_AGENT_BASE_URL") or os.environ.get("KIMI_BASE_URL")
                    or "https://api.moonshot.ai/v1")
        _API_KEY = (os.environ.get("KIMI_AGENT_API_KEY") or os.environ.get("KIMI_API_KEY")
                    or os.environ.get("MOONSHOT_API_KEY") or "")
        _MODEL_NAME = os.environ.get("CR_MODEL_NAME", "kimi-k2.7-code")
        _VLLM_KWARGS = False
        _TEMPERATURE = float(os.environ.get("CR_TEMPERATURE", "1.0"))  # k2.7-code accepts only temp=1
        if not _API_KEY:
            raise SystemExit("provider=kimi but no KIMI_AGENT_API_KEY / KIMI_API_KEY set")
    else:  # local
        ENDPOINT = os.environ.get("CR_MODEL_ENDPOINT", "http://192.168.40.105:31111/v1")
        _API_KEY = ""
        _MODEL_NAME = os.environ.get("CR_MODEL_NAME", "")
        _VLLM_KWARGS = True
        _TEMPERATURE = float(os.environ.get("CR_TEMPERATURE", "0.0"))

EDIT_RE = re.compile(
    r"###\s*EDIT\s+(?P<path>\S+)\s*\n<{5,}\s*SEARCH\s*\n(?P<search>.*?)\n={5,}\s*\n(?P<replace>.*?)\n>{5,}\s*REPLACE",
    re.DOTALL)


def model_id() -> str:
    if _MODEL_NAME:
        return _MODEL_NAME
    with urllib.request.urlopen(f"{ENDPOINT}/models", timeout=15) as r:
        return json.load(r)["data"][0]["id"]


def chat_messages(model: str, messages: list[dict], max_tokens: int = 2048, retries: int = 3) -> str:
    """A reasoning model at temperature=1 (kimi-k2.7-code) sometimes spends the whole output budget on
    reasoning and returns EMPTY content (finish_reason=length). Because temperature=1 varies the reasoning
    length, a retry usually lands a non-empty completion — so retry on empty rather than let a fluke read as
    a model failure. Returns the last completion (possibly empty) if all retries come back empty."""
    payload = {"model": model, "messages": messages, "temperature": _TEMPERATURE, "max_tokens": max_tokens}
    if _VLLM_KWARGS:
        payload["chat_template_kwargs"] = {"enable_thinking": False}  # vLLM-only; Moonshot rejects it
    headers = {"Content-Type": "application/json"}
    if _API_KEY:
        headers["Authorization"] = f"Bearer {_API_KEY}"
    req = urllib.request.Request(f"{ENDPOINT}/chat/completions", data=json.dumps(payload).encode(),
                                 headers=headers)
    content = ""
    for _ in range(max(1, retries)):
        with urllib.request.urlopen(req, timeout=300) as r:
            content = json.load(r)["choices"][0]["message"].get("content") or ""
        if content.strip():
            return content
    return content


def chat(model: str, system: str, user: str, max_tokens: int = 2048) -> str:
    return chat_messages(model, [{"role": "system", "content": system},
                                 {"role": "user", "content": user}], max_tokens)


SYSTEM = (
    "You are a precise code-editing agent. You are given a task and ONLY a subset of the repository's code as "
    "context. The code shown in CONTEXT is the CURRENT state of the repository (the change has NOT been made "
    "yet). Make the change. Output ONLY edits, no prose, in this exact format, one block per edit:\n"
    "### EDIT <relative/file/path.py>\n<<<<<<< SEARCH\n<exact existing lines to find>\n=======\n"
    "<replacement lines>\n>>>>>>> REPLACE\n"
    "The SEARCH block MUST be copied VERBATIM from the current code shown in CONTEXT — do NOT put your intended "
    "changes in SEARCH; put them only in the REPLACE half. Edit every place the change requires, even across "
    "files. If a needed file is not in your context, still emit your best-guess edit for it.")

# Anchored application (P4): instead of a free-text SEARCH the agent must echo, it names a SYMBOL from its
# context and returns that symbol's WHOLE new body; we splice it in by the symbol's known line range. This
# removes the string-echo confound, so resolution reflects whether the agent can make the CHANGE — and an
# arm can only anchor-edit symbols it was actually given, so a missed closure member still fails (the whole
# point). A symbol the arm never retrieved cannot be edited: that IS the retrieval signal.
SYSTEM_ANCHORED = (
    "You are a precise code-editing agent. You are given a task and ONLY a subset of the repository's code as "
    "context — each unit is labelled `### SYMBOL <id>` with its CURRENT source (the change is NOT made yet). "
    "To edit a unit, output its FULL new source, at the same indentation, in this exact format:\n"
    "### SYMBOL <id>\n```python\n<the complete new body of that symbol>\n```\n"
    "Emit one block per symbol you change; change every symbol the task requires (definition AND every call "
    "site). You may ONLY edit symbols shown in your context. Output ONLY these blocks, no prose.")

# Tolerant of model formatting: any/no language tag after the fence, and trailing prose after the symbol id.
ANCHOR_RE = re.compile(
    r"###\s*SYMBOL\s+(?P<id>[^\s`]+)[^\n]*\n```[A-Za-z0-9]*[ \t]*\n(?P<body>.*?)\n```", re.DOTALL)


# retrieved units carry token cost + location but not source text; re-read from the index at prompt time.
_SRC_INDEX: dict[str, str] = {}


def _source_of(u: RetrievedUnit) -> str:
    return _SRC_INDEX.get(u.symbol, "# (source unavailable)")


def build_user(task_desc: str, ctx: RetrievedContext, mode: str) -> str:
    if mode == "anchored":
        parts = [f"TASK:\n{task_desc}\n", "CONTEXT (label each editable unit; edit by SYMBOL id):"]
        for u in ctx.units:
            parts.append(f"\n### SYMBOL {u.symbol}\n```python\n{_source_of(u)}\n```")
        return "\n".join(parts)
    parts = [f"TASK:\n{task_desc}\n", "CONTEXT (the only code you have been given):"]
    for u in ctx.units:
        loc = f"{u.file}" + (f":{u.line_range[0]}-{u.line_range[1]}" if u.line_range else "")
        parts.append(f"\n--- {u.symbol}  [{loc}] ---\n{_source_of(u)}")
    return "\n".join(parts)


def apply_anchored(wt: str, text: str) -> dict:
    """Splice each `### SYMBOL <id>` block over that symbol's CURRENT line range. We re-index the worktree on
    every call, so ranges stay valid across multiple localize-verify rounds (round-1 edits shift line numbers;
    a stale base-tree range would corrupt the file). A symbol not in the worktree index is unappliable — the
    retrieval signal. Multiple edits to one file are applied bottom-up so earlier line numbers stay valid."""
    from resolver import index_repo
    wt_idx = index_repo(wt)
    blocks = list(ANCHOR_RE.finditer(text))
    by_file: dict[str, list[tuple[int, int, str]]] = {}
    applied, failed = 0, []
    for m in blocks:
        sid = m.group("id").strip()
        sym = wt_idx.get(sid)
        if sym is None:
            failed.append(f"unknown-symbol:{sid}")
            continue
        by_file.setdefault(sym.file, []).append((sym.start, sym.end, m.group("body")))
    for path, edits in by_file.items():   # path is already absolute inside the worktree
        lines = open(path, encoding="utf-8").read().splitlines()
        for start, end, body in sorted(edits, key=lambda e: -e[0]):   # bottom-up
            lines[start - 1:end] = body.splitlines()
            applied += 1
        open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    return {"edits_emitted": len(blocks), "applied": applied, "failed": failed}


def make_worktree(repo: str, commit: str) -> str:
    wt = tempfile.mkdtemp(prefix="cr_resolve_")
    subprocess.run(["git", "-C", repo, "worktree", "add", "--detach", wt, commit],
                   check=True, capture_output=True)
    return wt


def remove_worktree(repo: str, wt: str) -> None:
    subprocess.run(["git", "-C", repo, "worktree", "remove", "--force", wt], capture_output=True)
    shutil.rmtree(wt, ignore_errors=True)


def _safe_join(wt: str, rel: str) -> str | None:
    """Resolve `rel` inside the worktree, refusing anything that escapes it. Critical: a model sometimes
    emits an ABSOLUTE path, and os.path.join(wt, "/abs") silently discards wt — which would write into the
    REAL repo. Return None for absolute paths or `..` traversal so such an edit is recorded, never applied."""
    p = os.path.normpath(os.path.join(wt, rel))
    root = os.path.normpath(wt)
    return p if (p == root or p.startswith(root + os.sep)) else None


def apply_edits(wt: str, text: str) -> dict:
    edits = list(EDIT_RE.finditer(text))
    applied, failed = 0, []
    for m in edits:
        rel = m.group("path").strip()
        path = _safe_join(wt, rel)
        search, replace = m.group("search"), m.group("replace")
        if path is None:
            failed.append(f"escapes-worktree:{rel}")
            continue
        if not os.path.isfile(path):
            failed.append(f"no-file:{m.group('path')}")
            continue
        src = open(path, encoding="utf-8").read()
        if search not in src:
            failed.append(f"no-match:{m.group('path')}")
            continue
        open(path, "w", encoding="utf-8").write(src.replace(search, replace, 1))
        applied += 1
    return {"edits_emitted": len(edits), "applied": applied, "failed": failed}


def run_tests(wt: str, targets: list[str]) -> dict:
    env = dict(os.environ, PYTHONPATH=wt)
    # --tb=short: the traceback names the exact file:line that still needs editing — that IS the "localize"
    # signal the verify loop feeds back to the agent.
    p = subprocess.run([VENV_PY, "-m", "pytest", *targets, "-q", "--tb=short", "-p", "no:cacheprovider"],
                       cwd=wt, env=env, capture_output=True, text=True, timeout=300)
    return {"passed": p.returncode == 0, "returncode": p.returncode,
            "tail": (p.stdout + p.stderr)[-2400:]}


# The localize-verify feedback: the agent gets the FAILING TEST OUTPUT (which a real coding agent always has)
# and must fix the remaining sites — WITHOUT being handed any new code. If a failing site is a symbol the arm
# never retrieved, the agent can't see it to fix it, so the failure persists: that is the retrieval signal,
# preserved. It does not leak the oracle (test output is legitimately available).
VERIFY_FEEDBACK = (
    "Your edits were applied. Running the test suite now reports:\n\n{tail}\n\n"
    "Fix the REMAINING problems. The tracebacks point to the exact file:line locations that still need "
    "editing — typically call sites that must be updated to match a changed signature. Output ONLY more edit "
    "blocks in the SAME format, editing ONLY symbols shown in your original context. If nothing needs "
    "changing, output nothing.")


def _drop_acceptance(wt: str, acceptance_files: dict[str, str]) -> None:
    for rel, srcpath in acceptance_files.items():
        dst = os.path.join(wt, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(srcpath, dst)


def _attempt(repo: str, commit: str, user: str, acceptance_files: dict[str, str],
             must_not_regress: list[str], model: str, mode: str, verify_rounds: int) -> dict:
    """One resolution attempt with a localize-verify loop: edit -> run tests -> feed failures back -> edit
    again, up to `verify_rounds` extra rounds. The worktree persists across rounds so edits accumulate."""
    wt = make_worktree(repo, commit)
    try:
        system = SYSTEM_ANCHORED if mode == "anchored" else SYSTEM
        apply_fn = apply_anchored if mode == "anchored" else apply_edits
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        edit_text = chat_messages(model, messages, max_tokens=EDIT_MAX_TOKENS)
        appl = apply_fn(wt, edit_text)
        _drop_acceptance(wt, acceptance_files)
        targets = list(acceptance_files) + must_not_regress
        tests = run_tests(wt, targets)
        applied_total, rounds = appl["applied"], 0
        # An empty completion (a reasoning model starved of output budget) can't continue and must never be
        # appended as an assistant turn — some APIs reject empty content — so stop the loop instead.
        while not tests["passed"] and rounds < verify_rounds and edit_text.strip():
            rounds += 1
            messages += [{"role": "assistant", "content": edit_text},
                         {"role": "user", "content": VERIFY_FEEDBACK.format(tail=tests["tail"])}]
            edit_text = chat_messages(model, messages, max_tokens=EDIT_MAX_TOKENS)
            if not edit_text.strip():
                break
            applied_total += apply_fn(wt, edit_text)["applied"]
            tests = run_tests(wt, targets)
        return {"resolved": tests["passed"], "edits_emitted": appl["edits_emitted"],
                "applied": applied_total, "failed": appl["failed"], "verify_rounds": rounds, "tests": tests}
    finally:
        remove_worktree(repo, wt)


def resolve_arm(repo: str, commit: str, task_desc: str, ctx: RetrievedContext,
                acceptance_files: dict[str, str], must_not_regress: list[str],
                model: str, attempts: int = 2, mode: str = "anchored", verify_rounds: int = 2) -> dict:
    """pass@k: vLLM greedy is not perfectly deterministic, so a single shot is noisy. Run `attempts` shots;
    `resolved` if ANY passes. Records every attempt so the failure modes stay visible."""
    user = build_user(task_desc, ctx, mode)
    tries = [_attempt(repo, commit, user, acceptance_files, must_not_regress, model, mode, verify_rounds)
             for _ in range(attempts)]
    best = next((t for t in tries if t["resolved"]), tries[0])
    return {"resolved": any(t["resolved"] for t in tries), "attempts": tries,
            "edits_emitted": best["edits_emitted"], "applied": best["applied"], "failed": best["failed"],
            "tests": best["tests"], "materialized_tokens": ctx.materialized_tokens}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/mnt/backup/projects/agentic-os")
    ap.add_argument("--commit", default="f8ce79c")
    ap.add_argument("--task", default="signature_change_v1")
    ap.add_argument("--budget", type=int, default=4000)
    ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("--mode", choices=["anchored", "search_replace"], default="anchored")
    ap.add_argument("--verify-rounds", type=int, default=2, dest="verify_rounds",
                    help="localize-verify follow-up rounds (0 = single-shot, no test feedback)")
    ap.add_argument("--provider", choices=["local", "kimi"], default="local",
                    help="local = Qwen3.8-27B on the proxmox GPU; kimi = Moonshot Kimi via cloud API")
    ap.add_argument("--out", default=os.path.join(HERE, "results_resolution.json"))
    a = ap.parse_args()
    _configure(a.provider)

    index = index_repo(a.repo)
    _SRC_INDEX.update({sid: s.source for sid, s in index.items()})

    # locate the task's oracle
    task_path = None
    for fn in os.listdir(os.path.join(HERE, "oracle", "tasks")):
        p = os.path.join(HERE, "oracle", "tasks", fn)
        if fn.endswith(".yaml") and yaml.safe_load(open(p)).get("task_id") == a.task:
            task_path = p
            break
    if not task_path:
        raise SystemExit(f"no oracle for task {a.task}")
    raw = yaml.safe_load(open(task_path))
    oracle = Oracle.load(task_path)
    o_ret = oracle.for_retrieval()
    task_desc = raw.get("description", "")

    # frozen acceptance tests: match resolution.targets (tests/*.py) to files in oracle/acceptance/
    acc_dir = os.path.join(HERE, "oracle", "acceptance")
    acceptance_files = {}
    for tgt in raw.get("resolution", {}).get("targets", []):
        cand = os.path.join(acc_dir, os.path.basename(tgt))
        if os.path.isfile(cand):
            acceptance_files[tgt] = cand
    must_not_regress = (raw.get("resolution", {}).get("must_not_regress", "") or "").split()

    query = task_desc
    graph = build_call_graph(index)
    arms = {
        "existing_cr_hybrid": existing_cr_hybrid_arm(query, index, a.budget),
        "graph_closure": graph_closure_arm(o_ret.seed_symbols, index, graph, a.budget),
        "graph_plus_hybrid": graph_plus_hybrid_arm(o_ret.seed_symbols, query, index, graph, a.budget),
        "oracle_closure": oracle_closure_arm(o_ret, index),
    }
    model = model_id()
    print(f"model={model}  endpoint={ENDPOINT}  task={a.task}  budget={a.budget}")
    print(f"acceptance={list(acceptance_files)}  must_not_regress={must_not_regress}\n")

    out = {"task": a.task, "declaration_hash": oracle.declaration_hash, "model": model,
           "endpoint": ENDPOINT, "budget_tokens": a.budget, "mode": a.mode, "attempts_per_arm": a.attempts,
           "verify_rounds": a.verify_rounds, "arms": {}}
    print(f"mode={a.mode}  verify_rounds={a.verify_rounds}")
    for name, ctx in arms.items():
        res = resolve_arm(a.repo, a.commit, task_desc, ctx, acceptance_files, must_not_regress,
                          model, a.attempts, a.mode, a.verify_rounds)
        mt = res["materialized_tokens"]
        res["resolution_efficiency"] = (1.0 if res["resolved"] else 0.0) / (mt / 1000) if mt else None
        out["arms"][name] = res
        n_ok = sum(t["resolved"] for t in res["attempts"])
        vr = max((t.get("verify_rounds", 0) for t in res["attempts"]), default=0)
        print(f"{name:16} resolved={res['resolved']!s:5} ({n_ok}/{a.attempts})  edits={res['edits_emitted']} "
              f"applied={res['applied']} failed={res['failed']}  verify_rounds={vr}  mat_tok={mt}  "
              f"rc={res['tests']['returncode']}")
    json.dump(out, open(a.out, "w"), indent=2, sort_keys=True)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
