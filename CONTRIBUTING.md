# Contributing

Issues and pull requests are welcome: <https://github.com/redevops-io/redevops-integrations/issues>.

- Each slice lives under `integrations/<slice>/` with its own README, `conformance.py` and `results/` bundles. Shared code is in `integrations/common/`.
- A comparison result must use one of the existing classifications (`PARITY`, `FRAMEWORK_NATIVE_ADVANTAGE`, `REDEVOPS_DELTA`, `EXPECTED_IMPLEMENTATION_DIFFERENCE`, `BUG`; see `integrations/common/classification.py`). A check that does not run the framework is a `RUNTIME_PROPERTY`, never one of those five; `tests/test_conformance_tally.py` enforces this and keeps each README tally equal to the committed bundles.
- Headline numbers in READMEs must point at a committed `results/*.json` bundle; otherwise label them as illustrative.
- Never commit credentials; CI runs a secret scan (`.github/workflows/secret-scan.yml`).
- Contributions are accepted under Apache-2.0 (`LICENSE`).

Security reports: see [SECURITY.md](SECURITY.md).
