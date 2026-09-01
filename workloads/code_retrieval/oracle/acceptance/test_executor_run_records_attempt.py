"""FROZEN acceptance test for the `signature_change` family (resolution criterion).

Passes ONLY if `Executor.run` gained an `attempt` parameter AND every runtime call site threads it — i.e. the
whole contract-caller closure was changed, not just the definition. It is deliberately STRUCTURAL (inspect +
source scan) rather than behavioural, so it is robust to whatever edit style the agent chooses and cannot be
satisfied by touching the definition alone (which is the entire point of the closure thesis).

Dropped into the isolated worktree's `tests/` by apply_loop.py before pytest runs; never committed to the
repo under test.
"""
import inspect
import re

import agentic_os.mission.executor as ex
import agentic_os.mission.runtime as rt


def test_executor_run_gained_attempt_param():
    sig = inspect.signature(ex.Executor.run)
    assert "attempt" in sig.parameters, \
        f"Executor.run must accept an `attempt` parameter; got {list(sig.parameters)}"


def test_every_runtime_call_site_threads_attempt():
    src = inspect.getsource(rt)
    calls = re.findall(r"\.run\(\s*node\s*,\s*inputs[^)]*\)", src)
    assert calls, "no executor `.run(node, inputs ...)` call sites found in runtime.py"
    missing = [c for c in calls if "attempt" not in c]
    assert not missing, f"call site(s) do not thread `attempt` (closure incomplete): {missing}"
