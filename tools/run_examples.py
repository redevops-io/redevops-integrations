#!/usr/bin/env python3
"""Run slice examples for real and hold their findings against the committed bundles.

    python tools/run_examples.py integrations/crewai/examples/governed_delegation.py [more examples ...]

Each example is executed with the interpreter running this script (so with whatever framework SDKs are installed). It must exit 0, write its
bundle, report no BUG, and reproduce the committed bundle's findings (experiment, metric, value, classification). A replay finding must come
from a replay that ran (`consistent=True`). An example that says NOT RUN (a missing model key, for instance) fails here: it is not a pass.
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
KEYS = ("experiment", "metric", "with_redevops", "classification")


def committed(path: pathlib.Path) -> dict:
    out = subprocess.run(["git", "show", f"HEAD:{path.relative_to(ROOT).as_posix()}"], cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"{path}: no committed bundle to compare with")
    return json.loads(out.stdout)


def check(example: str) -> list:
    ex = (ROOT / example).resolve()
    run = subprocess.run([sys.executable, ex.name], cwd=ex.parent, capture_output=True, text=True, timeout=900)
    text = run.stdout + run.stderr
    if run.returncode != 0:
        return [f"exit {run.returncode}: {text.strip().splitlines()[-1][:300] if text.strip() else 'no output'}"]
    m = re.search(r"wrote (\S+\.json)", text)
    if not m:
        return ["the example wrote no bundle"]
    bundle_path = pathlib.Path(m.group(1)).resolve()
    new, old = json.loads(bundle_path.read_text()), committed(bundle_path)
    problems = []
    got = [{k: f.get(k) for k in KEYS} for f in new.get("findings", [])]
    want = [{k: f.get(k) for k in KEYS} for f in old.get("findings", [])]
    if not got:
        problems.append("the bundle has no findings")
    if got != want:
        problems.append(f"findings differ from the committed bundle: got {got} want {want}")
    for f in new.get("findings", []):
        if f.get("classification") == "BUG":
            problems.append(f"BUG: {f.get('experiment')}")
        if "replay" in str(f.get("experiment", "")).lower() and "consistent=True" not in str(f.get("note", "")):
            problems.append(f"{f.get('experiment')}: the replay did not run to a consistent result ({f.get('note')})")
    return problems


def main(argv: list) -> int:
    if not argv:
        print(__doc__)
        return 2
    failed = 0
    for example in argv:
        problems = check(example)
        print(("FAIL " if problems else "PASS ") + example)
        for p in problems:
            print("     " + p)
        failed += bool(problems)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
