# Status

| | |
|---|---|
| Maturity | reference implementations; results Internally reproduced (committed bundles, not externally reproduced) |
| Version | none: no tags or releases |
| Last git activity | 2026-10-13 (`git log -1 --format=%cs`; clone is shallow) |
| Activity | active: 8 commits between 2026-07-15 and 2026-10-15 (GitHub API) |
| Mission SDK pin | `redevops-mission@v0.3.9` in `requirements.txt` (the latest `mission-sdk` tag, 2026-11-24; it pins kernel `agentic-os` `v0.4.2`). Re-run on it on 2026-11-24 with unchanged findings: the governed examples of aws, azure, digitalocean, google and crewai (CrewAI 1.15.44). Not re-run: nvidia and pydanticai (they call a model and need its key), llamaindex (needs `redevops_rag`, not in `requirements.txt`). The other bundles were produced on an older SDK and say so only through their `adapter_commit` |
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
