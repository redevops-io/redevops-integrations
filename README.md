# ReDevOps Integrations

[![NVIDIA Inception](https://img.shields.io/badge/NVIDIA-Inception%20Program%20Member-76B900.svg)](https://www.nvidia.com/en-us/startups/)

**What it does.** Reference implementations of the ReDevOps Mission Runtime wrapped around eight agent frameworks and clouds
(NVIDIA NeMo Agent Toolkit, PydanticAI, LlamaIndex, CrewAI, Azure Semantic Kernel, Google ADK/A2A, AWS Strands, DigitalOcean GenAI).
Each slice keeps the framework's native agent loop and tests what the runtime adds around it, then classifies every
native-vs-native+ReDevOps comparison. It is a conformance harness, not an adapter library and not a leaderboard.

**Who it's for.** People evaluating how the Mission SDK composes with an agent framework or cloud, and reviewers who want the
frozen acceptance bundles. Its direct dependency is [`mission-sdk`](https://github.com/redevops-io/mission-sdk) (`redevops-mission`).

**Maturity: reference, internally reproduced.** No tags or releases. Result figures below come from committed JSON
acceptance bundles produced by ReDevOps; they have not been reproduced externally. See [STATUS.md](STATUS.md).

**Ownership, overlap and support status.** Owned by ReDevOps. Status: **active** (last commit 2026-10-13; 8 commits between
2026-07-15 and 2026-10-15 per the GitHub API). Primary consumer: none found in the clones checked; the only reference outside this
repo is `agentic-os/scripts/check_pin_currency.py`, which tracks this repo's pins, and the website (not audited). Not the connector SDK:
that is [`redevops-connectors`](https://github.com/redevops-io/redevops-connectors). See [OVERLAP](#overlap). Questions and bug reports:
GitHub Issues ([CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md)).

**3-minute example.** The fastest slices to run are the standalone ones (e.g. `crewai`); they need the framework SDK from
`requirements.txt` and an OpenAI-compatible endpoint. These are the commands from "Setup" and "Run" below; they were not
executed while writing this README (they install large dependencies).

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cd integrations/crewai && python conformance.py     # prints the slice's classification tally
```

**Evidence.** Committed acceptance bundles at `integrations/<slice>/results/*.json` (linked per slice in the table below;
framework version, model ids, per-finding classification, result digest). Label: **Internally reproduced** (not re-run for this README).
Spot check: `nv_b_gdpr_structural.json` contains 0.913 and `li_a_code_closure.json` contains 0.693.

**Limitations.** Version pin: `requirements.txt` pins `redevops-mission@v0.3.4`; the latest `mission-sdk` tag is `v0.3.7`
(`git ls-remote --tags`, 2026-10-15). Which SDK version produced each bundle is not stated here; bumping the pin is a config change not made in this PR.
The LI-B bundle comes from a benchmark harness that is **not included** in this repo (see Reproduction notes). Some runs need an
external dataset or code repo. No cloud credentials are used; cloud identity is modelled as policy.

Live write-ups, architecture diagrams and benchmark tables: **[redevops.io/integrations](https://redevops.io/integrations)**.
The Mission SDK these build on: **[github.com/redevops-io/mission-sdk](https://github.com/redevops-io/mission-sdk)**.

Every slice runs a **real** framework agent, **locally** — the four hyperscaler slices need **no cloud credentials**
(the SDKs point at any OpenAI-compatible endpoint; cloud identity is modelled as policy). One provider-neutral
adapter contract (`integrations/common/`) is shared by all eight.

## The eight integrations

| Slice | Framework / cloud | What ReDevOps adds (the boundary tested) | Headline | Committed results (Internally reproduced) |
|---|---|---|---|---|
| [nvidia](integrations/nvidia) | NVIDIA NeMo Agent Toolkit | closure-aware context + governed Mission | SciFact 0.800→0.867 · GDPR 0.412→0.913 · 0 replay dups | [nv_a_scifact_closure](integrations/nvidia/results/nv_a_scifact_closure.json), [nv_b_gdpr_structural](integrations/nvidia/results/nv_b_gdpr_structural.json), [nv_d_governed_tool](integrations/nvidia/results/nv_d_governed_tool.json) |
| [pydanticai](integrations/pydanticai) | PydanticAI | verification/governance beyond the typed boundary | 4 schema-valid outputs refused · 0 replay dups | [pa_a_governed_strategy](integrations/pydanticai/results/pa_a_governed_strategy.json), [pa_d_typed_verification](integrations/pydanticai/results/pa_d_typed_verification.json), [pa_e_closure_verdict](integrations/pydanticai/results/pa_e_closure_verdict.json) |
| [llamaindex](integrations/llamaindex) | LlamaIndex | the retriever as one representation, composed with closure | code closure 0.320→0.693 · 6/6 unsupported answers abstained | [li_a_code_closure](integrations/llamaindex/results/li_a_code_closure.json), [li_b_longcontext_answer](integrations/llamaindex/results/li_b_longcontext_answer.json), [li_d_governed_retrieval](integrations/llamaindex/results/li_d_governed_retrieval.json) |
| [crewai](integrations/crewai) | CrewAI | authority that narrows across delegation | 3/3 authority-widening actions denied · no pooling | [cr_a_delegation_authority](integrations/crewai/results/cr_a_delegation_authority.json), [cr_b_governed_delegation](integrations/crewai/results/cr_b_governed_delegation.json), [cr_c_crew_decision](integrations/crewai/results/cr_c_crew_decision.json) |
| [azure](integrations/azure) | Azure Semantic Kernel | compliance gate + Entra-composing authority (deny-wins) | 3/3 non-compliant deploys denied · 0 replay dups | [az_a_compliance_gated_deploy](integrations/azure/results/az_a_compliance_gated_deploy.json), [az_b_governed_deploy](integrations/azure/results/az_b_governed_deploy.json) |
| [google](integrations/google) | Google ADK / A2A | chain-wide authority + provenance across a multi-hop mesh | 2/2 transitive escalations denied · 2/2 unprovenanced excluded | [go_a_mesh_authority](integrations/google/results/go_a_mesh_authority.json), [go_b_governed_research](integrations/google/results/go_b_governed_research.json), [go_c_research_mesh](integrations/google/results/go_c_research_mesh.json) |
| [aws](integrations/aws) | AWS Strands Agents | Mission authority composed with IAM (deny-wins) + shadow | 4/4 policy-denied actions denied · 500 households, 0 side effects | [aw_a_deny_wins_iam](integrations/aws/results/aw_a_deny_wins_iam.json), [aw_b_shadow_promotion](integrations/aws/results/aw_b_shadow_promotion.json), [aw_c_real_wealth_agent](integrations/aws/results/aw_c_real_wealth_agent.json) |
| [digitalocean](integrations/digitalocean) | DigitalOcean GenAI | one shared runtime, tenant-isolated | 3/3 cross-tenant accesses denied · machinery 5 once vs 15 | [do_a_shared_runtime](integrations/digitalocean/results/do_a_shared_runtime.json), [do_b_governed_tenant](integrations/digitalocean/results/do_b_governed_tenant.json), [do_c_three_apps](integrations/digitalocean/results/do_c_three_apps.json) |

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

## OVERLAP

Several ReDevOps repositories sound alike. This table is identical in `redevops-connectors`, `redevops-integrations` and `agent-harness`.

| Repo | Owns | Does not own | Licence file (see its own repo) |
|---|---|---|---|
| [`redevops-connectors`](https://github.com/redevops-io/redevops-connectors) | The `IntegrationAdapter` contract, OAuth/credential-reference plumbing, and 14 provider adapters (Gmail, Google Calendar, HubSpot, Stripe, Polar, Slack, WhatsApp Cloud API, WhatsApp via WAHA, Klaviyo, Postiz, Ayrshare, Blotato, Apollo, Web). Tested against canned fixtures only. | Governance decisions, Missions, framework benchmarks. | `LICENSE.md` (AGPL-3.0-or-later with Commons Clause) |
| [`redevops-integrations`](https://github.com/redevops-io/redevops-integrations) | Eight framework/cloud conformance slices (NVIDIA NeMo, PydanticAI, LlamaIndex, CrewAI, Azure, Google, AWS, DigitalOcean) that wrap a real framework agent with the Mission SDK and classify native vs native+ReDevOps results. | The connector adapter contract, the Mission SDK itself. | Apache-2.0 |
| [`agent-harness`](https://github.com/redevops-io/agent-harness) | Single-step agent building blocks: LLM client, tool registry, approval flow, sandbox, guardrails, eval harness. | Mission lifecycle, provider integrations, context planning. | AGPL-3.0 text (`LICENSE`) |
| [`mission-sdk`](https://github.com/redevops-io/mission-sdk) | The `redevops-mission` package and `rdo` CLI: the developer-facing way to author and run governed Missions. | Provider adapters, framework conformance slices. | Apache-2.0 |
| [`agentic-os`](https://github.com/redevops-io/agentic-os) | The control plane and Mission kernel, plus the reference apps; it is the main consumer of `redevops-connectors` (optional `connectors` / `projects` extras). | Connector contract source, SDK packaging. | See that repo |

Common confusions: `redevops-connectors` is the live-adapter SDK, `redevops-integrations` is a conformance harness (not an adapter library), and `agent-harness` is not a Mission runtime. Do not `pip install agent-harness` from PyPI: that name belongs to an unrelated third-party project.
