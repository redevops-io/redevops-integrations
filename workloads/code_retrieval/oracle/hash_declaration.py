"""Freeze a task's REQUIRED-CLOSURE declaration before its reference patch exists.

The oracle must not be derived from the gold diff (that leaks the developer's implementation choices into
the target). Instead a task declares its seed symbols and required dependency CLASSES up front; this tool
canonicalizes that declaration (excluding the hash field) and stamps a sha256, so the target is provably
fixed before the reference patch is written.

    python hash_declaration.py tasks/<task>.yaml            # print the hash
    python hash_declaration.py tasks/<task>.yaml --write     # stamp declaration_hash into the file
"""
import hashlib
import json
import sys

import yaml  # PyYAML


def canonical(decl: dict) -> bytes:
    d = {k: v for k, v in decl.items() if k != "declaration_hash"}
    return json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def compute(path: str) -> str:
    with open(path) as f:
        decl = yaml.safe_load(f)
    return "sha256:" + hashlib.sha256(canonical(decl)).hexdigest()[:32]


if __name__ == "__main__":
    path = sys.argv[1]
    h = compute(path)
    if "--write" in sys.argv:
        with open(path) as f:
            lines = f.read().splitlines()
        lines = [ln for ln in lines if not ln.startswith("declaration_hash:")]
        lines.append(f'declaration_hash: "{h}"')
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print("stamped", path, h)
    else:
        print(h)
