# Status

| | |
|---|---|
| Maturity | reference implementations; results Internally reproduced (committed bundles, not externally reproduced) |
| Version | none: no tags or releases |
| Last git activity | 2026-10-13 (`git log -1 --format=%cs`; clone is shallow) |
| Activity | active: 8 commits between 2026-07-15 and 2026-10-15 (GitHub API) |
| Mission SDK pin | `redevops-mission@v0.3.4` in `requirements.txt`; latest `mission-sdk` tag is `v0.3.7` (`git ls-remote --tags`, 2026-10-15) |
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
