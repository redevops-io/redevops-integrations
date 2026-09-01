"""Symbol resolver (Phase 0, build step 5 prerequisite).

Indexes a repo into `module:symbol` units via the Python AST — precise for our Python repos, no external
deps. This is used ONLY to (a) give arms a corpus of retrievable units and (b) report which symbols an arm
surfaced (with file, line range, token cost) so the scorer can run. It is NOT the proposed plugin — it does
no graph expansion; it's plain symbol indexing shared by every arm.
"""
from __future__ import annotations

import ast
import os
import textwrap
from dataclasses import dataclass, field


@dataclass
class Sym:
    id: str          # "module:name" or "module:Class.method"
    file: str
    start: int
    end: int
    source: str
    tokens: int      # ~len/4 estimate
    kind: str        # func | method | class


def _module(root: str, path: str) -> str:
    rel = os.path.relpath(path, root)
    rel = rel[:-3] if rel.endswith(".py") else rel
    mod = rel.replace(os.sep, ".")
    if mod.endswith(".__init__"):
        mod = mod[: -len(".__init__")]
    return mod


def _add(syms: dict[str, Sym], sid: str, path: str, node, lines: list[str], kind: str) -> None:
    start = node.lineno
    end = getattr(node, "end_lineno", start) or start
    source = "\n".join(lines[start - 1: end])
    syms[sid] = Sym(sid, path, start, end, source, max(1, len(source) // 4), kind)


def index_repo(root: str, subdirs: tuple[str, ...] = ("agentic_os",)) -> dict[str, Sym]:
    syms: dict[str, Sym] = {}
    for sub in subdirs:
        base = os.path.join(root, sub)
        for dp, dns, files in os.walk(base):
            dns[:] = [d for d in dns if d != "__pycache__"]
            for fn in files:
                if not fn.endswith(".py"):
                    continue
                path = os.path.join(dp, fn)
                mod = _module(root, path)
                try:
                    src = open(path, encoding="utf-8").read()
                    tree = ast.parse(src)
                except (SyntaxError, UnicodeDecodeError):
                    continue
                lines = src.splitlines()
                for node in tree.body:
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        _add(syms, f"{mod}:{node.name}", path, node, lines, "func")
                    elif isinstance(node, ast.ClassDef):
                        _add(syms, f"{mod}:{node.name}", path, node, lines, "class")
                        for m in node.body:
                            if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                _add(syms, f"{mod}:{node.name}.{m.name}", path, m, lines, "method")
    return syms


@dataclass
class CallGraph:
    """Approximate static call graph over the symbol index. `callees[s]` = symbols s calls; `callers[s]` =
    symbols that call s. Resolution is intentionally simple and precise-for-Python: `self.m(...)` -> the same
    class's method, a bare `f(...)` -> a module-level function in the same module. `obj.m(...)` on an
    arbitrary receiver is NOT resolved (unknowable statically without types) — so this is a high-precision,
    partial-recall graph, which is exactly what a seed->closure traversal wants (few false edges)."""
    callees: dict[str, set[str]] = field(default_factory=dict)
    callers: dict[str, set[str]] = field(default_factory=dict)


def build_call_graph(index: dict[str, Sym], instantiation_edges: bool = True) -> CallGraph:
    """Static call graph. Edge types (all high-precision):
    - CALLS: `self.m(...)` -> same-class method; bare `f(...)` -> module-level function.
    - INSTANTIATES (opt-in, default on): bare `Foo(...)` where `Foo` is the UNIQUE name of an indexed class ->
      an edge to that class. This recovers the callers that construct a type (the graph's original blind spot
      on type-seeded changes), while staying precise by only resolving unambiguous class names.
    Unresolved `obj.m(...)` on an arbitrary receiver is still skipped (unknowable without type inference)."""
    g = CallGraph()
    mod_funcs: dict[str, dict[str, str]] = {}
    cls_methods: dict[tuple[str, str], dict[str, str]] = {}
    class_by_name: dict[str, str | None] = {}   # unique class name -> sid; None marks an ambiguous name
    for sid, s in index.items():
        mod, _, rest = sid.partition(":")
        if "." in rest:
            cls, _, meth = rest.partition(".")
            cls_methods.setdefault((mod, cls), {})[meth] = sid
        elif s.kind == "func":
            mod_funcs.setdefault(mod, {})[rest] = sid
        elif s.kind == "class":
            class_by_name[rest] = None if rest in class_by_name else sid

    for sid, s in index.items():
        mod, _, rest = sid.partition(":")
        cls = rest.split(".")[0] if "." in rest else None
        try:
            tree = ast.parse(textwrap.dedent(s.source))
        except (SyntaxError, ValueError):
            continue
        callees: set[str] = set()
        for n in ast.walk(tree):
            if not isinstance(n, ast.Call):
                continue
            f = n.func
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "self" and cls:
                tgt = cls_methods.get((mod, cls), {}).get(f.attr)
                if tgt and tgt != sid:
                    callees.add(tgt)
            elif isinstance(f, ast.Name):
                tgt = mod_funcs.get(mod, {}).get(f.id)
                if tgt and tgt != sid:
                    callees.add(tgt)
                elif instantiation_edges:
                    cls_tgt = class_by_name.get(f.id)   # unique class name -> INSTANTIATES edge
                    if cls_tgt and cls_tgt != sid:
                        callees.add(cls_tgt)
        if callees:
            g.callees[sid] = callees
            for c in callees:
                g.callers.setdefault(c, set()).add(sid)
    return g
