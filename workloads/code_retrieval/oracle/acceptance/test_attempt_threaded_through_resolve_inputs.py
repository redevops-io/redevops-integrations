"""FROZEN acceptance test for the `transitive_thread_param` family (resolution criterion).

Passes ONLY if an optional `attempt` parameter was threaded down the WHOLE chain: added to `_approval_edit`
(optional), added to `_resolve_inputs` which passes it to `_approval_edit`, and passed at BOTH dispatch call
sites (`_execute`, `_execute_wave`) where they call `_resolve_inputs`. A patch that stops at the seed + its
direct caller — the lexical/dense failure mode — leaves the two transitive dispatch calls un-threaded and
fails `test_dispatch_methods_thread_attempt`. Structural (inspect + source scan), robust to edit style.
"""
import inspect
import re

import agentic_os.mission.runtime as rt

MR = rt.MissionRuntime


def test_approval_edit_gained_optional_attempt():
    p = inspect.signature(MR._approval_edit).parameters
    assert "attempt" in p, f"_approval_edit must accept `attempt`; got {list(p)}"
    assert p["attempt"].default is not inspect.Parameter.empty, "`attempt` must be optional (defaulted)"


def test_resolve_inputs_threads_attempt_to_approval_edit():
    assert "attempt" in inspect.signature(MR._resolve_inputs).parameters, \
        "_resolve_inputs must accept `attempt`"
    calls = re.findall(r"self\._approval_edit\([^)]*\)", inspect.getsource(MR._resolve_inputs))
    assert calls, "no self._approval_edit(...) call found in _resolve_inputs"
    assert all("attempt" in c for c in calls), f"_resolve_inputs must thread attempt into _approval_edit: {calls}"


def test_dispatch_methods_thread_attempt_to_resolve_inputs():
    calls = re.findall(r"self\._resolve_inputs\([^)]*\)", inspect.getsource(rt))
    assert calls, "no self._resolve_inputs(...) call sites found"
    missing = [c for c in calls if "attempt" not in c]
    assert not missing, f"dispatch call site(s) do not thread attempt (transitive closure incomplete): {missing}"
