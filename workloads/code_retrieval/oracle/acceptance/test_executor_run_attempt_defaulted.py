"""FROZEN acceptance test for the `signature_change_compat` family (resolution criterion).

Passes ONLY if `Executor.run` gained an **optional** (defaulted) `attempt` parameter AND every runtime call
site threads it. The default is what makes the change backward-compatible: `must_not_regress` includes tests
that call `Executor.run(node, inputs)` directly, so a *required* param would break them. Structural
(inspect + source scan) so it is robust to edit style and cannot be satisfied by touching the definition
alone.

Dropped into the isolated worktree's `tests/` by apply_loop.py before pytest runs; never committed.
"""
import inspect
import re

import agentic_os.mission.executor as ex
import agentic_os.mission.runtime as rt


def test_attempt_param_is_optional_with_default():
    params = inspect.signature(ex.Executor.run).parameters
    assert "attempt" in params, \
        f"Executor.run must accept an `attempt` parameter; got {list(params)}"
    assert params["attempt"].default is not inspect.Parameter.empty, \
        "`attempt` must be OPTIONAL (have a default) so existing direct callers keep working"


def test_every_runtime_call_site_threads_attempt():
    src = inspect.getsource(rt)
    calls = re.findall(r"\.run\(\s*node\s*,\s*inputs[^)]*\)", src)
    assert calls, "no executor `.run(node, inputs ...)` call sites found in runtime.py"
    missing = [c for c in calls if "attempt" not in c]
    assert not missing, f"call site(s) do not thread `attempt` (closure incomplete): {missing}"
