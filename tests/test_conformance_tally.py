"""The published conformance tallies are what the committed result bundles and the conformance scripts produce, and a check that runs no framework
is never counted as a framework result.

    python -m pytest tests/test_conformance_tally.py        (standard library and pytest only; runs no framework and makes no network call)
"""
import json
import pathlib
import re
ORDER = ("REDEVOPS_DELTA", "PARITY", "FRAMEWORK_NATIVE_ADVANTAGE", "EXPECTED_IMPLEMENTATION_DIFFERENCE", "BUG", "RUNTIME_PROPERTY")
def computed(slice_dir):
    t = {}
    for f in sorted((slice_dir / "results").glob("*.json")) if (slice_dir / "results").is_dir() else []:
        for x in json.loads(f.read_text()).get("findings", []):
            t[x["classification"]] = t.get(x["classification"], 0) + 1
    src = (slice_dir / "conformance.py").read_text()
    body = src[:src.index("def main")]
    for cls in ORDER:
        n = len(re.findall(r"classification=ResultClass\.%s\b" % cls, body))
        if n: t[cls] = t.get(cls, 0) + n
    return t
def stated(slice_dir):
    line = next(l for l in (slice_dir / "README.md").read_text().splitlines() if l.startswith("conformance"))
    return {m.group(2): int(m.group(1)) for m in re.finditer(r"(\d+) ([A-Z_]+)", line)}

ROOT = pathlib.Path(__file__).resolve().parents[1] / "integrations"
SLICES = sorted(d for d in ROOT.iterdir() if (d / "conformance.py").exists())


def test_there_are_slices_to_check():
    assert len(SLICES) >= 8


def test_each_readme_tally_equals_the_bundles_plus_the_script():
    for d in SLICES:
        c = {k: v for k, v in computed(d).items() if v}
        s = {k: v for k, v in stated(d).items() if v}
        assert c == s, f"{d.name}: README says {s}, the bundles and conformance.py give {c}"


def test_a_finding_built_without_running_the_framework_is_never_a_redevops_delta():
    """Findings constructed inside conformance.py (before main() runs any example) exercise the runtime alone or assert the shape of a span tree the
    script itself built. They may be a RUNTIME_PROPERTY or an architectural EXPECTED_IMPLEMENTATION_DIFFERENCE, never a measured comparison."""
    for d in SLICES:
        src = (d / "conformance.py").read_text()
        body = src[:src.index("def main")]
        for cls in ("REDEVOPS_DELTA", "PARITY", "FRAMEWORK_NATIVE_ADVANTAGE"):
            assert not re.search(r"ResultClass\.%s\b" % cls, body), f"{d.name}: a hand-built finding is classified {cls}"


def test_every_tally_prints_runtime_properties_on_their_own_line():
    for d in SLICES:
        assert '"RUNTIME_PROPERTY")' in (d / "conformance.py").read_text(), d.name
