# ReDevOps Integrations

[![NVIDIA Inception](https://img.shields.io/badge/NVIDIA-Inception%20Program%20Member-76B900.svg)](https://www.nvidia.com/en-us/startups/)

Reference implementations of the **ReDevOps Mission Runtime** wrapped around the agent frameworks and clouds you
already use. Each integration keeps the framework's native agent loop and adds the production runtime properties
*around* it — closure-aware context, authority, approval, replay, verification and governance — then classifies
every native-vs-native+ReDevOps comparison. **Conformance, not a scorecard.**

Live write-ups, architecture diagrams and benchmark tables: **[redevops.io/integrations](https://redevops.io/integrations)**.
The Mission SDK these build on: **[github.com/redevops-io/mission-sdk](https://github.com/redevops-io/mission-sdk)**.

Every slice runs a **real** framework agent, **locally** — the four hyperscaler slices need **no cloud credentials**
(the SDKs point at any OpenAI-compatible endpoint; cloud identity is modelled as policy). One provider-neutral
adapter contract (`integrations/common/`) is shared by all eight.

## The eight integrations

| Slice | Framework / cloud | What ReDevOps adds (the boundary tested) | Headline |
|---|---|---|---|
| [nvidia](integrations/nvidia) | NVIDIA NeMo Agent Toolkit | closure-aware context + governed Mission | SciFact 0.800→0.867 · GDPR 0.412→0.913 · 0 replay dups |
| [pydanticai](integrations/pydanticai) | PydanticAI | verification/governance beyond the typed boundary | 4 schema-valid outputs refused · 0 replay dups |
| [llamaindex](integrations/llamaindex) | LlamaIndex | the retriever as one representation, composed with closure | code closure 0.320→0.693 · 6/6 unsupported answers abstained |
| [crewai](integrations/crewai) | CrewAI | authority that narrows across delegation | 3/3 authority-widening actions denied · no pooling |
| [azure](integrations/azure) | Azure Semantic Kernel | compliance gate + Entra-composing authority (deny-wins) | 3/3 non-compliant deploys denied · 0 replay dups |
| [google](integrations/google) | Google ADK / A2A | chain-wide authority + provenance across a multi-hop mesh | 2/2 transitive escalations denied · 2/2 unprovenanced excluded |
| [aws](integrations/aws) | AWS Strands Agents | Mission authority composed with IAM (deny-wins) + shadow | 4/4 policy-denied actions denied · 500 households, 0 side effects |
| [digitalocean](integrations/digitalocean) | DigitalOcean GenAI | one shared runtime, tenant-isolated | 3/3 cross-tenant accesses denied · machinery 5 once vs 15 |

Each slice's `README.md` has its own thesis, run commands, expected output and honesty line. Each writes
content-addressed **acceptance bundles** to `results/` (framework version, model ids, per-finding classification,
result digest) — these are committed, so the measured evidence is public even where a run needs extra setup.

## Result classification

Every comparison is one of: `PARITY` · `FRAMEWORK_NATIVE_ADVANTAGE` · `REDEVOPS_DELTA` ·
`EXPECTED_IMPLEMENTATION_DIFFERENCE` · `BUG` (`integrations/common/classification.py`). The point is a faithful
map of where the runtime adds value, not a leaderboard.

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # Mission SDK + the framework SDKs you want
```

The Mission SDK (`redevops-mission`) is the one dependency every slice needs. Install the framework SDKs for the
slices you want to run (see `requirements.txt`). Model calls go to any OpenAI-compatible endpoint via
`OPENAI_API_KEY` / `OPENAI_BASE_URL`.

**Azure runs in a dedicated venv** — Semantic Kernel pins pydantic/openai below the floor the other slices share:

```bash
uv venv integrations/azure/.venv --python 3.12
VIRTUAL_ENV=integrations/azure/.venv uv pip install semantic-kernel \
  "redevops-mission @ git+https://github.com/redevops-io/mission-sdk"
```

## Run

```bash
cd integrations/<slice>
python conformance.py            # runs the slice's experiments + prints the classification tally
# (azure: use .venv/bin/python)
```

## Reproduction notes (honest)

- **Fully standalone (no data, no external repo):** all of **aws · azure · crewai · digitalocean · google**, plus
  the governed / typed / replay examples of nvidia, pydanticai and llamaindex. `git clone`, install, run.
- **Need a public dataset fetch:** nvidia `scifact_closure` / `gdpr_structural` and pydanticai `closure_verdict`
  read SciFact / GDPR — see [`workloads/thirddomain/data/README.md`](workloads/thirddomain/data/README.md).
- **Need an external code repo:** llamaindex `code_closure` (LI-A) indexes a target repository via `--repo`
  (the measured run used the pinned `agentic-os` tree); point it at any Python repo to reproduce the mechanism.
- **Need a long-context corpus:** llamaindex `longcontext_answer` (LI-B) reads a LiveRAG-style haystack via a
  benchmark harness not included here; the committed `results/li_b_*.json` bundle captures the measured run.

In every case the code and the frozen acceptance bundles are public here; only a few runs need the extra input.

## License

Apache-2.0 (`LICENSE`). Framework and cloud names are trademarks of their respective owners; these are independent
integrations, not products or endorsements, and use no cloud services.
