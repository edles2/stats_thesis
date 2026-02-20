# stats_radiotherapy

Oral medicine thesis statistics module.

Portfolio-ready analysis module for comparing automated dental dose methods with a manual reference.

## 60-second overview
- Context: statistics support module for an oral medicine thesis.
- Core goal: evaluate method agreement (bias, absolute error, Spearman correlation) by dental zones.
- Main code lives in `analysis/`.
- Private raw workbook is excluded from Git for privacy.

## Start here
1. `analysis/data.py` for Excel-to-tidy transformation.
2. `analysis/analyse.py` for pair matching and summary metrics.
3. `analysis/streamlit.py` for interactive exploration.
4. `analysis/smoke.py` for quick synthetic smoke check.

## What is in `analysis/`
- `data.py`: parses sector-based workbook structure into long-format rows.
- `analyse.py`: builds manual-vs-method pairs and computes per-zone summaries.
- `streamlit.py`: interactive dashboard over analysis outputs.
- `requirements.txt`: Python environment dependencies.
- `smoke.py`: no-private-data smoke run using synthetic inputs.
- `private_data/README.md`: where local private workbook should be placed.

## Reproduce in <5 minutes
Environment method: `requirements.txt` (pip/venv).

```bash
cd analysis
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python smoke.py
```

Optional app run (requires local private workbook):

```bash
cd analysis
streamlit run streamlit.py
```

## Data privacy
- Excluded from repository: private/raw thesis workbook and any patient-identifiable source data.
- Expected local-only path for private data: `analysis/private_data/thesis_source_data.xlsx`.
- Public smoke check uses synthetic data only (`analysis/smoke.py`) and does not require secrets or external services.

## How results are produced
- Raw workbook -> tidy data (`build_tidy_dataframe`).
- Tidy data -> manual/method tooth-level pairs (`prepare_pairs`).
- Pairs -> per-method/per-zone summary table (`compute_zone_summary`).
