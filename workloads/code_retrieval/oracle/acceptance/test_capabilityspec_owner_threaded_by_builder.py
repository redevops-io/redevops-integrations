"""FROZEN acceptance test for the `type_field_add` family (resolution criterion).

Passes ONLY if `CapabilitySpec` gained an optional `owner` field AND the `capability(...)` builder gained an
optional `owner` parameter that it threads into the `CapabilitySpec(...)` it constructs. The default keeps
every other constructor working (must_not_regress). Structural (dataclass fields + inspect + source scan).
"""
import dataclasses
import inspect

import agentic_os.mission.operator_sdk as sdk
import agentic_os.mission.types as types


def test_capabilityspec_gained_optional_owner_field():
    fmap = {f.name: f for f in dataclasses.fields(types.CapabilitySpec)}
    assert "owner" in fmap, f"CapabilitySpec must have an `owner` field; got {sorted(fmap)}"
    f = fmap["owner"]
    assert f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING, \
        "`owner` must be optional (have a default) so existing constructions keep working"


def test_builder_threads_owner_into_constructor():
    params = inspect.signature(sdk.capability).parameters
    assert "owner" in params, f"capability() must accept an `owner` parameter; got {list(params)}"
    src = inspect.getsource(sdk.capability)
    assert "owner=" in src, "capability() must pass `owner=` into the CapabilitySpec(...) it constructs"
