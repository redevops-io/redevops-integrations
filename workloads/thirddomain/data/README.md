# Closure-resolution datasets (not redistributed here)

The scientific and regulatory closure examples read two public datasets that are **not bundled** in this repo (to
keep dataset licensing clean). Fetch them into this directory to reproduce those examples locally; the frozen
`results/*.json` acceptance bundles already capture the measured runs regardless.

Which examples need them:

- `integrations/nvidia/examples/scifact_closure.py` (NV-A) · `integrations/pydanticai/examples/closure_verdict.py`
  (PA-E) → **SciFact** (`allenai/scifact`, CC BY-NC) → `data/scifact/`
- `integrations/nvidia/examples/gdpr_structural.py` (NV-B) → **GDPR cross-reference graph** (GDPRtEXT) → `data/gdpr/`

Everything else — all of AWS, Azure, CrewAI, DigitalOcean and Google, plus the governed/typed/replay examples of
NVIDIA, PydanticAI and LlamaIndex — runs with no dataset at all.
