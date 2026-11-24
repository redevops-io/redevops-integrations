# Status

| | |
|---|---|
| Maturity | reference implementations; results Internally reproduced (committed bundles, not externally reproduced) |
| Version | none: no tags or releases |
| Last git activity | 2026-10-13 (`git log -1 --format=%cs`; clone is shallow) |
| Activity | active: 8 commits between 2026-07-15 and 2026-10-15 (GitHub API) |
| Mission SDK pin | `redevops-mission@v0.3.9` in `requirements.txt` (kernel `agentic-os` `v0.4.2`); `redevops-rag@v0.4.0` for the LlamaIndex slice. On 2026-11-24 all eight governed examples were re-run on it with the real frameworks: crewai 1.15.44, llama-index-core 0.14.26, pydantic-ai-slim 2.31.1 and nvidia-nat 1.6.0, the last two against a self-hosted OpenAI-compatible endpoint (`CR_OPENAI_MODEL=Qwen3.6-35B-A3B`, recorded in their bundles); aws, azure, digitalocean and google need no framework SDK for this example. CI (`framework-integrations`) re-runs crewai and the four cloud examples on every change and holds their findings against the committed bundles. The non-governed bundles (retrieval and closure benchmarks) were produced on an older SDK |
| Licence | Apache-2.0 (`LICENSE`) |

## Supported

- Eight slices (nvidia, pydanticai, llamaindex, crewai, azure, google, aws, digitalocean), each with a `conformance.py` and committed `results/*.json` bundles.
- Local runs against any OpenAI-compatible endpoint; the hyperscaler slices need no cloud credentials.

## Not supported / not established

- Not an adapter library (that is `redevops-connectors`) and not a leaderboard.
- Results were not re-run for this documentation change; no external reproduction exists.
- The LI-B (`llamaindex` long-context) bundle came from a benchmark harness that is not included here.
- Some runs need an external dataset (SciFact/GDPR) or a target code repo (LlamaIndex LI-A); see the README's reproduction notes.
- `workloads/code_retrieval/README.md` refers to `docs/` plan/result files that are not in this repo.

## Known issues

- Dependency chain: this repo is Apache-2.0 and requires `redevops-mission`, which in turn depends on `agentic-os`.
- Pin is behind the latest Mission SDK tag (see above).

Related repositories and what each owns: see OVERLAP in [README.md](README.md#overlap).
