"""Acceptance bundle (both plans, Section 22) — the reproducibility record every integration run emits."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field

from classification import Finding


def _git_sha(path: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _hash(obj) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


@dataclass
class AcceptanceBundle:
    framework: str
    framework_version: str
    adapter_commit: str = field(default_factory=lambda: _git_sha(os.path.dirname(os.path.dirname(__file__))))
    mission_sdk_version: str = ""
    context_runtime_version: str = ""
    dataset_hashes: dict = field(default_factory=dict)
    model_ids: list = field(default_factory=list)
    config: dict = field(default_factory=dict)
    findings: list = field(default_factory=list)              # list[Finding]
    tests: dict = field(default_factory=dict)

    def add(self, f: Finding):
        self.findings.append(f)

    def result_digest(self) -> str:
        return _hash([asdict(f) if isinstance(f, Finding) else f for f in self.findings])

    def to_json(self) -> dict:
        d = asdict(self)
        d["findings"] = [asdict(f) if isinstance(f, Finding) else f for f in self.findings]
        for x in d["findings"]:
            if isinstance(x.get("classification"), object) and hasattr(x["classification"], "value"):
                x["classification"] = x["classification"].value
        d["result_digest"] = self.result_digest()
        return d

    def write(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_json(), fh, indent=2, default=lambda o: getattr(o, "value", str(o)))
        return path
